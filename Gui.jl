"""
    Gui

Интерактивное окно для визуализации динамических систем на основе GLMakie.

Единственная публичная функция — `run_app()`. Открывает окно 1440×900 с тремя
вкладками (3D-траектория, фазовый портрет, временны́е ряды) и панелью управления.

Любое изменение слайдера или меню мгновенно пересчитывает траекторию через
реактивные `Observable` — перерисовка происходит автоматически без явных вызовов.

Зависимости: `Models` → определения систем, `Solvers` → интегрирование, `Analysis` → статистика.

## Важное упрощение интерфейса

Панель управления показывает только **три** слайдера параметров системы
(`Параметр 1/2/3`), хотя `LotkaVolterraSystem` и `DoublePendulumSystem` принимают
по 4 параметра. Четвёртый параметр в обоих случаях зафиксирован константой
(см. `SYSTEM_SPECS` и `build_system` ниже) — это сознательный компромисс между
простотой интерфейса и полнотой управления, а не недосмотр. Подписи слайдеров
обновляются под выбранную систему, чтобы было понятно, что означает каждый
ползунок.
"""
module Gui

using GLMakie

include("Models.jl");   using .Models
include("Solvers.jl");  using .Solvers
include("Analysis.jl"); using .Analysis

export run_app

# =============================================================================
# КОНСТАНТЫ ИНТЕРФЕЙСА
# =============================================================================

const WINDOW_W   = 1440
const WINDOW_H   = 900
const DT_DEFAULT = 0.01
const T_DEFAULT  = 50.0

const SYSTEMS = [
    "Аттрактор Лоренца",
    "Хищник-жертва (Лотка-Вольтерра)",
    "Осциллятор Ван дер Поля",
    "Аттрактор Росслера",
    "Двойной маятник",
]

# -----------------------------------------------------------------------------
# ЕДИНЫЙ ИСТОЧНИК ПРАВДЫ ДЛЯ ПАРАМЕТРОВ СИСТЕМ
#
# Раньше параметры по умолчанию были продублированы в трёх независимых местах
# (make_system, обработчик system_menu.selection, реактивная lift-цепочка).
# Любое расхождение между ними молча ломало GUI: слайдер показывал бы одно
# значение, а реально использовалось другое. Здесь — одна таблица, из которой
# берут данные все три места.
#
# `labels`  — подписи трёх слайдеров параметров для этой системы
# `values`  — их стартовые значения (в том же порядке, что и labels)
# `extra`   — параметры системы, НЕ управляемые слайдерами (см. докстринг модуля)
# -----------------------------------------------------------------------------

struct SystemSpec
    labels :: NTuple{3, String}
    values :: NTuple{3, Float64}
    extra  :: NTuple{N, Float64} where N
end

const SYSTEM_SPECS = Dict(
    1 => SystemSpec(("σ (Прандтль)", "ρ (Рэлей)", "β (геометрия)"),
                     (10.0, 28.0, 8/3), ()),
    2 => SystemSpec(("α (рост жертв)", "β (охота)", "γ (гибель хищ.)"),
                     (1.0, 0.1, 1.5), (0.075,)),          # δ зафиксирована
    3 => SystemSpec(("μ (нелинейность)", "—", "—"),
                     (1.0, 1.0, 1.0), ()),                 # только 1-й слайдер активен
    4 => SystemSpec(("a (раскрутка)", "b (выброс по z)", "c (порог)"),
                     (0.2, 0.2, 5.7), ()),
    5 => SystemSpec(("m₁ (масса 1)", "m₂ (масса 2)", "—"),
                     (1.0, 1.0, 1.0), (1.0, 1.0)),         # L1, L2 зафиксированы
)

"""
    build_system(idx, p1, p2, p3) -> система

Собирает систему по индексу меню `idx` (см. `SYSTEMS`) и значениям трёх
слайдеров параметров. Параметры, не управляемые слайдерами, берутся из
`SYSTEM_SPECS[idx].extra` — единственного места, где они заданы.
"""
function build_system(idx::Int, p1::Float64, p2::Float64, p3::Float64)
    extra = SYSTEM_SPECS[idx].extra
    if idx == 1; return Models.LorenzSystem(p1, p2, p3)
    elseif idx == 2; return Models.LotkaVolterraSystem(p1, p2, p3, extra[1])
    elseif idx == 3; return Models.VanDerPolSystem(p1)
    elseif idx == 4; return Models.RosslerSystem(p1, p2, p3)
    else; return Models.DoublePendulumSystem(p1, p2, extra[1], extra[2])
    end
end

# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================

"""
    make_system(idx) -> система

Создаёт экземпляр системы по индексу меню с параметрами по умолчанию из
`SYSTEM_SPECS`. `idx` соответствует порядку в `SYSTEMS`: 1=Лоренц,
2=Лотка-Вольтерра, 3=Ван дер Поль, 4=Росслер, 5=двойной маятник.
"""
function make_system(idx::Int)
    p1, p2, p3 = SYSTEM_SPECS[idx].values
    return build_system(idx, p1, p2, p3)
end

"""
    compute_speeds(u_results, dt) -> Vector{Float64}

Вычисляет скорость движения в фазовом пространстве в каждой точке траектории.
Скорость в точке `i`: `‖u[i+1] − u[i]‖ / dt`. Последнее значение дублируется
из предпоследнего, чтобы длина совпадала с длиной `u_results`.
"""
function compute_speeds(u_results::Vector{Vector{Float64}}, dt::Float64)
    n = length(u_results)
    speeds = zeros(Float64, n)
    for i in 1:(n-1)
        diff = u_results[i+1] .- u_results[i]
        speeds[i] = sqrt(sum(x^2 for x in diff)) / dt
    end
    speeds[end] = speeds[end-1]
    return speeds
end

"""
    normalize_to_unit(v) -> Vector{Float64}

Нормирует вектор `v` к диапазону `[0, 1]`. Если все значения одинаковы
(диапазон < 1e-10), возвращает нулевой вектор той же длины.
"""
function normalize_to_unit(v::Vector{Float64})
    vmin, vmax = minimum(v), maximum(v)
    r = vmax - vmin
    return r < 1e-10 ? zeros(Float64, length(v)) : (v .- vmin) ./ r
end

"""
    run_solver(sys, u0, tspan, dt) -> u_results

Запускает `Solvers.solve_rk4` и возвращает только вектор состояний
(без `t_steps` и `SolverStats`). Используется как вычислительное ядро
в реактивной `lift`-цепочке GUI.
"""
function run_solver(sys, u0, tspan, dt)
    _, u_res, _ = Solvers.solve_rk4(sys, u0, tspan, dt)
    return u_res
end

# =============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# =============================================================================

"""
    run_app()

Запускает интерактивное окно симулятора.

Создаёт окно GLMakie 1440×900 с двумя панелями:

**Левая панель — графики (три вкладки):**
- *3D траектория* — пространственный вид; полезен для трёхмерных систем (Лоренц, Росслер)
- *Фазовый портрет* — проекция на плоскость (X, Y)
- *Временны́е ряды* — X(t) и Y(t) на одном графике с легендой

**Правая панель — управление:**
- выбор системы из меню
- три слайдера параметров системы
- два слайдера начальных условий (u₀[1], u₀[2])
- слайдеры шага `dt` и времени интегрирования `T`
- переключатель цветовой карты (по скорости / однородный)
- кнопка сброса к параметрам по умолчанию
- статистика траектории (число точек, длина пути, диапазон X)
- кнопка сохранения текущего вида в PNG

Все элементы управления связаны с вычислением через `Observable` и `lift` —
изменение любого слайдера немедленно пересчитывает траекторию и обновляет графики.
"""
function run_app()

    # -------------------------------------------------------------------------
    # РЕАКТИВНОЕ СОСТОЯНИЕ
    # -------------------------------------------------------------------------
    system_idx_obs   = Observable(1)
    dt_obs           = Observable(DT_DEFAULT)
    T_obs            = Observable(T_DEFAULT)
    param1_obs       = Observable(SYSTEM_SPECS[1].values[1])
    param2_obs       = Observable(SYSTEM_SPECS[1].values[2])
    param3_obs       = Observable(SYSTEM_SPECS[1].values[3])
    u0_x_obs         = Observable(1.0)
    u0_y_obs         = Observable(1.0)
    use_colormap_obs = Observable(true)
    active_tab_obs   = Observable(1)   # 1=3D  2=Phase  3=Time

    # Подписи слайдеров параметров — ОТДЕЛЬНЫЕ Observable{String}, не атрибуты
    # самого Slider. У типа Slider в Makie нет изменяемого поля `label`: подпись,
    # переданная в SliderGrid(...; label = "...") создаётся как статичный Label
    # внутри SliderGrid и недоступна для последующего изменения через
    # sliders[i].label. Поэтому подписи рисуются вручную через отдельные Label
    # с lift-привязкой текста (см. ниже), по тому же паттерну, что и заголовок
    # 3D-графика.
    param_label1_obs = Observable(SYSTEM_SPECS[1].labels[1])
    param_label2_obs = Observable(SYSTEM_SPECS[1].labels[2])
    param_label3_obs = Observable(SYSTEM_SPECS[1].labels[3])

    # -------------------------------------------------------------------------
    # ВЫЧИСЛЕНИЕ ТРАЕКТОРИИ
    # -------------------------------------------------------------------------
    u_results_obs = lift(
        system_idx_obs, param1_obs, param2_obs, param3_obs,
        u0_x_obs, u0_y_obs, dt_obs, T_obs
    ) do idx, p1, p2, p3, x0, y0, dt, T
        sys = build_system(idx, p1, p2, p3)
        u0_base = Models.default_u0(sys)
        u0_base[1] = x0
        length(u0_base) >= 2 && (u0_base[2] = y0)
        return Base.invokelatest(run_solver, sys, u0_base, (0.0, T), dt)
    end

    speeds_obs = lift(u_results_obs) do u
        Float32.(normalize_to_unit(compute_speeds(u, dt_obs[])))
    end

    line_colors_obs = lift(speeds_obs, use_colormap_obs) do sp, use_cm
        use_cm ? sp : fill(0.5f0, length(sp))
    end

    # -------------------------------------------------------------------------
    # ОКНО И ГЛАВНАЯ СЕТКА
    # -------------------------------------------------------------------------
    fig = Figure(size = (WINDOW_W, WINDOW_H), backgroundcolor = :gray10)

    # Создаём обе панели — только после этого можно задавать colsize!
    left_panel  = fig[1, 1] = GridLayout()
    right_panel = fig[1, 2] = GridLayout()
    colsize!(fig.layout, 1, Relative(0.74))
    colsize!(fig.layout, 2, Fixed(360))

    # -------------------------------------------------------------------------
    # ЛЕВАЯ ПАНЕЛЬ
    # Строка 1: три вложенных GridLayout (по одному на вкладку), стек в [1,1..3]
    # Строка 2: кнопки-вкладки
    # Вкладка переключается через colsize! на строке 1 left_panel
    # -------------------------------------------------------------------------

    # Три контейнера в одной строке — активный получает Relative(1), остальные Fixed(0)
    gl3d   = left_panel[1, 1] = GridLayout()
    gl2d   = left_panel[1, 2] = GridLayout()
    gltime = left_panel[1, 3] = GridLayout()

    # Сразу скрываем неактивные вкладки
    colsize!(left_panel, 1, Relative(1.0))
    colsize!(left_panel, 2, Fixed(0))
    colsize!(left_panel, 3, Fixed(0))

    # Переключатель вкладок
    on(active_tab_obs) do tab
        colsize!(left_panel, 1, tab == 1 ? Relative(1.0) : Fixed(0))
        colsize!(left_panel, 2, tab == 2 ? Relative(1.0) : Fixed(0))
        colsize!(left_panel, 3, tab == 3 ? Relative(1.0) : Fixed(0))
    end

    # -- Оси — каждая в своём контейнере --
    ax3d = Axis3(
        gl3d[1, 1],
        title           = lift(i -> "Система: " * SYSTEMS[i], system_idx_obs),
        xlabel          = "X", ylabel = "Y", zlabel = "Z",
        aspect          = (1, 1, 0.8),
        titlecolor      = :white,
        xlabelcolor     = :gray70, ylabelcolor  = :gray70, zlabelcolor  = :gray70,
        xticklabelcolor = :gray70, yticklabelcolor = :gray70, zticklabelcolor = :gray70,
        backgroundcolor = (:black, 0.3),
    )

    ax2d = Axis(
        gl2d[1, 1],
        title           = "",
        xlabel          = "X", ylabel = "Y",
        titlecolor      = :white,
        xlabelcolor     = :gray70, ylabelcolor     = :gray70,
        xticklabelcolor = :gray70, yticklabelcolor = :gray70,
        backgroundcolor = (:black, 0.3),
    )

    ax_time = Axis(
        gltime[1, 1],
        title           = "",
        xlabel          = "Время t", ylabel = "Значение",
        titlecolor      = :white,
        xlabelcolor     = :gray70, ylabelcolor     = :gray70,
        xticklabelcolor = :gray70, yticklabelcolor = :gray70,
        backgroundcolor = (:black, 0.3),
    )

    # Скрываем неактивные вкладки полностью
    function set_tab_visible(ax, v)
        ax.blockscene.visible[] = v
        ax.scene.visible[]      = v
    end
    set_tab_visible(ax2d,    false)
    set_tab_visible(ax_time, false)

    on(active_tab_obs) do tab
        set_tab_visible(ax3d,    tab == 1)
        set_tab_visible(ax2d,    tab == 2)
        set_tab_visible(ax_time, tab == 3)
    end

    # -- Кнопки вкладок (строка 2 left_panel) --
    btn_row = left_panel[2, 1:3] = GridLayout()
    tab_titles = ["3D траектория", "Фазовый портрет", "Временные ряды"]
    for i in 1:3
        btn = Button(btn_row[1, i],
                     label       = tab_titles[i],
                     buttoncolor = :gray25,
                     labelcolor  = :white,
                     height      = 36)
        on(btn.clicks) do _; active_tab_obs[] = i; end
    end

    rowsize!(left_panel, 1, Relative(1.0))
    rowsize!(left_panel, 2, Fixed(44))

    # -------------------------------------------------------------------------
    # ГРАФИКИ
    # -------------------------------------------------------------------------

    # --- 3D ---
    x3 = lift(u -> Float32[p[1] for p in u], u_results_obs)
    y3 = lift(u -> Float32[p[2] for p in u], u_results_obs)
    z3 = lift(u_results_obs) do u
        length(u[1]) >= 3 ? Float32[p[3] for p in u] : zeros(Float32, length(u))
    end
    lines!(ax3d, x3, y3, z3, color = line_colors_obs, colormap = :plasma, linewidth = 0.8)
    on(u_results_obs) do _; autolimits!(ax3d); end

    # --- Фазовый портрет ---
    x2 = lift(u -> Float32[p[1] for p in u], u_results_obs)
    y2 = lift(u -> Float32[p[2] for p in u], u_results_obs)
    lines!(ax2d, x2, y2, color = line_colors_obs, colormap = :plasma, linewidth = 0.8)
    on(u_results_obs) do _; autolimits!(ax2d); end

    # --- Временные ряды ---
    t_axis = lift((u, dt) -> Float32.(range(0.0, step=dt, length=length(u))),
                  u_results_obs, dt_obs)
    x_time = lift(u -> Float32[p[1] for p in u], u_results_obs)
    y_time = lift(u -> Float32[p[2] for p in u], u_results_obs)
    lines!(ax_time, t_axis, x_time, color = :dodgerblue, linewidth = 1.2, label = "X(t)")
    lines!(ax_time, t_axis, y_time, color = :orangered,  linewidth = 1.2, label = "Y(t)")
    axislegend(ax_time, labelcolor = :white, backgroundcolor = :gray20)
    on(u_results_obs) do _; autolimits!(ax_time); end

    # -------------------------------------------------------------------------
    # ПРАВАЯ ПАНЕЛЬ: УПРАВЛЕНИЕ
    # -------------------------------------------------------------------------
    r = 0
    right_panel.default_rowgap = Fixed(8)

    # Заголовок
    r += 1
    Label(right_panel[r, 1:2], "УПРАВЛЕНИЕ",
          fontsize = 16, color = :white, font = :bold,
          halign = :center, tellwidth = false)

    # Система
    r += 1
    Label(right_panel[r, 1:2], "Система:",
          color = :gray80, halign = :left, tellwidth = false)
    r += 1
    system_menu = Menu(right_panel[r, 1:2],
                       options = SYSTEMS, default = SYSTEMS[1], tellwidth = true)
    # ВАЖНО: обработчик `on(system_menu.selection)` регистрируется ниже,
    # ПОСЛЕ определения param_sliders/sg_u0 — он ссылается на них.
    # GLMakie может прислать начальное событие selection сразу при создании
    # Menu, до того как остальные виджеты были бы определены, что приводило
    # бы к UndefVarError и тихо ломало переключение системы на старте.

    # Параметры системы
    #
    # Используются три отдельных Slider (а не SliderGrid) с собственными
    # Label-подписями, привязанными через lift к param_label{1,2,3}_obs.
    # SliderGrid здесь не подходит: его подписи — статичные Label, заданные
    # один раз при создании, без публичного способа изменить их позже
    # (см. комментарий у определения param_label1_obs выше).
    #
    # Каждая пара (подпись, слайдер) — во вложенном GridLayout (как и tgl_grid
    # ниже), чтобы длинная подпись не задавала ширину всей колонки 1
    # right_panel и не сжимала слайдер.
    r += 1
    Label(right_panel[r, 1:2], "Параметры системы:",
          color = :gray80, halign = :left, tellwidth = false)

    r += 1
    pg1 = right_panel[r, 1:2] = GridLayout()
    Label(pg1[1, 1], param_label1_obs, color = :gray80, halign = :left, width = 110, tellwidth = false)
    slider_p1 = Slider(pg1[1, 2], range = 0.01:0.01:30.0, startvalue = SYSTEM_SPECS[1].values[1])
    Label(pg1[1, 3], lift(v -> string(round(v, digits=3)), slider_p1.value),
          color = :gray70, halign = :right, width = 50, tellwidth = false)
    colgap!(pg1, 1, 8); colgap!(pg1, 2, 8)

    r += 1
    pg2 = right_panel[r, 1:2] = GridLayout()
    Label(pg2[1, 1], param_label2_obs, color = :gray80, halign = :left, width = 110, tellwidth = false)
    slider_p2 = Slider(pg2[1, 2], range = 0.01:0.1:50.0, startvalue = SYSTEM_SPECS[1].values[2])
    Label(pg2[1, 3], lift(v -> string(round(v, digits=3)), slider_p2.value),
          color = :gray70, halign = :right, width = 50, tellwidth = false)
    colgap!(pg2, 1, 8); colgap!(pg2, 2, 8)

    r += 1
    pg3 = right_panel[r, 1:2] = GridLayout()
    Label(pg3[1, 1], param_label3_obs, color = :gray80, halign = :left, width = 110, tellwidth = false)
    slider_p3 = Slider(pg3[1, 2], range = 0.01:0.01:10.0, startvalue = SYSTEM_SPECS[1].values[3])
    Label(pg3[1, 3], lift(v -> string(round(v, digits=3)), slider_p3.value),
          color = :gray70, halign = :right, width = 50, tellwidth = false)
    colgap!(pg3, 1, 8); colgap!(pg3, 2, 8)

    # Группируем для единообразного доступа ниже (set_close_to! и т.д.)
    param_sliders = (slider_p1, slider_p2, slider_p3)
    on(param_sliders[1].value) do v; param1_obs[] = v; end
    on(param_sliders[2].value) do v; param2_obs[] = v; end
    on(param_sliders[3].value) do v; param3_obs[] = v; end

    # Начальные условия
    r += 1
    Label(right_panel[r, 1:2], "Начальные условия:",
          color = :gray80, halign = :left, tellwidth = false)
    r += 1
    sg_u0 = SliderGrid(
        right_panel[r, 1:2],
        (label = "u₀[1]", range = -10.0:0.1:10.0, startvalue = 1.0),
        (label = "u₀[2]", range = -10.0:0.1:10.0, startvalue = 1.0),
        tellwidth = true
    )
    on(sg_u0.sliders[1].value) do v; u0_x_obs[] = v; end
    on(sg_u0.sliders[2].value) do v; u0_y_obs[] = v; end

    # Обработчик смены системы — определён здесь, а не сразу после Menu(...),
    # потому что ссылается на param_sliders и sg_u0 (см. комментарий выше).
    on(system_menu.selection) do sel
        idx = findfirst(==(sel), SYSTEMS)
        isnothing(idx) && return

        spec = SYSTEM_SPECS[idx]
        sys  = make_system(idx)
        u0   = Models.default_u0(sys)

        # Подписи (через Observable{String}) и значения слайдеров параметров —
        # из единой таблицы SYSTEM_SPECS, больше не дублируются здесь вручную.
        param_label1_obs[] = spec.labels[1]
        param_label2_obs[] = spec.labels[2]
        param_label3_obs[] = spec.labels[3]
        for i in 1:3
            set_close_to!(param_sliders[i], spec.values[i])
        end

        u0_x_obs[] = u0[1]
        u0_y_obs[] = length(u0) >= 2 ? u0[2] : 0.0
        system_idx_obs[] = idx
    end

    # Интегрирование
    r += 1
    Label(right_panel[r, 1:2], "Интегрирование:",
          color = :gray80, halign = :left, tellwidth = false)
    r += 1
    sg_solver = SliderGrid(
        right_panel[r, 1:2],
        (label = "Шаг dt",  range = 0.001:0.001:0.05, startvalue = DT_DEFAULT),
        (label = "Время T", range = 10.0:5.0:200.0,   startvalue = T_DEFAULT),
        tellwidth = true
    )
    on(sg_solver.sliders[1].value) do v; dt_obs[] = v; end
    on(sg_solver.sliders[2].value) do v; T_obs[]  = v; end

    # Colormap
    r += 1
    Label(right_panel[r, 1:2], "Визуализация:",
          color = :gray80, halign = :left, tellwidth = false)
    r += 1
    tgl_grid = right_panel[r, 1:2] = GridLayout()
    cm_toggle = Toggle(tgl_grid[1, 1], active = true)
    Label(tgl_grid[1, 2],
          lift(v -> v ? "Colormap: скорость" : "Цвет: однородный", cm_toggle.active),
          color = :gray80, halign = :left, tellwidth = false)
    colgap!(tgl_grid, 1, 8)
    on(cm_toggle.active) do val; use_colormap_obs[] = val; end

    # Сброс параметров
    r += 1
    reset_btn = Button(right_panel[r, 1:2],
                       label = "Сбросить параметры",
                       buttoncolor = :gray30, labelcolor = :white,
                       tellwidth = true, height = 34)
    on(reset_btn.clicks) do _
        idx  = system_idx_obs[]
        spec = SYSTEM_SPECS[idx]
        u0   = Models.default_u0(make_system(idx))

        for i in 1:3
            set_close_to!(param_sliders[i], spec.values[i])
        end
        set_close_to!(sg_u0.sliders[1], u0[1])
        set_close_to!(sg_u0.sliders[2], length(u0) >= 2 ? u0[2] : 0.0)
        set_close_to!(sg_solver.sliders[1], DT_DEFAULT)
        set_close_to!(sg_solver.sliders[2], T_DEFAULT)
    end

    # Статистика
    r += 1
    Label(right_panel[r, 1:2], "Статистика:",
          color = :gray80, halign = :left, tellwidth = false)
    r += 1
    info_label = Label(right_panel[r, 1:2], "...",
                       color = :gray70, fontsize = 11,
                       halign = :left, tellwidth = false,
                       justification = :left)
    on(u_results_obs) do u
        n    = length(u)
        dt   = dt_obs[]
        path = sum(sqrt(sum((u[i+1][d]-u[i][d])^2 for d in eachindex(u[i])))
                   for i in 1:(n-1))
        xs   = [p[1] for p in u]
        x_rng = round(maximum(xs) - minimum(xs), digits=1)
        info_label.text[] = "Точек: $n\nВремя: $(round((n-1)*dt, digits=1))\nПуть:  $(round(path, digits=1))\nΔX:    $x_rng"
    end

    # Экспорт PNG
    r += 1
    save_btn = Button(right_panel[r, 1:2],
                      label = "Сохранить PNG",
                      buttoncolor = :gray30, labelcolor = :white,
                      tellwidth = true, height = 34)
    on(save_btn.clicks) do _
        fname = replace("attractor_$(SYSTEMS[system_idx_obs[]]).png",
                        " " => "_", "(" => "", ")" => "", "-" => "")
        save(fname, fig)
        info_label.text[] = "Сохранено:\n$fname"
    end

    # -------------------------------------------------------------------------
    display(fig)
end

end # module Gui