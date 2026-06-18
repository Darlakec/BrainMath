"""
    Analysis

Инструменты для анализа траекторий динамических систем.

Основные функции:

- `lyapunov_exponent`     — показатель Ляпунова: λ > 0 означает хаос
- `trajectory_stats`      — статистика по готовой траектории (среднее, разброс, скорость, длина)
- `divergence_rate`       — скорость расхождения двух близких траекторий во времени
- `phase_space_density`   — гистограмма плотности в фазовом пространстве (первые две координаты)
- `conservation_check`    — дрейф интеграла движения как мера точности солвера
- `print_analysis_report` — сводный отчёт по системе в консоль
"""
module Analysis

using ..Models

export lyapunov_exponent, trajectory_stats, TrajectoryStats
export divergence_rate, phase_space_density, conservation_check
export print_analysis_report

# =============================================================================
# ПОКАЗАТЕЛЬ ЛЯПУНОВА
#
# МАТЕМАТИЧЕСКАЯ ИДЕЯ:
#   Берём два близких начальных условия: u₀ и u₀ + δu₀
#   где δu₀ — малое возмущение, ||δu₀|| = ε (обычно ε = 1e-8)
#
#   Интегрируем обе траектории параллельно.
#   Расстояние между ними со временем меняется как:
#
#       d(t) ≈ d(0) * e^(λt)
#
#   где λ — показатель Ляпунова.
#
#   λ > 0: траектории расходятся экспоненциально → ХАОС
#   λ = 0: траектории расходятся степенно → граница хаоса
#   λ < 0: траектории сближаются → устойчивое равновесие
#
# ПРОБЛЕМА ПРЯМОГО ВЫЧИСЛЕНИЯ:
#   Если просто интегрировать две траектории, расстояние быстро
#   достигает размера аттрактора и перестаёт расти. Мы потеряем
#   информацию об экспоненте.
#
# МЕТОД RENORMALIZATION (перенормировка):
#   Через каждые n_renorm шагов:
#   1. Вычислить расстояние d_curr = ||u - u_perturbed||
#   2. Накопить вклад: λ_sum += log(d_curr / ε)
#   3. Сбросить возмущение обратно к ε: u_perturbed = u + δu * (ε / d_curr)
#      (сохраняем направление δu, но нормируем длину обратно к ε)
#
#   Итоговый показатель:
#   λ = λ_sum / (n_renorm_steps * dt * n_renorm)   (в единицах 1/время)
#
# ПАРАМЕТРЫ:
#   n_steps     — число шагов интегрирования
#   dt          — шаг
#   n_renorm    — шагов между перенормировками (обычно 10–100)
#   epsilon     — начальный размер возмущения (обычно 1e-8)
# =============================================================================

"""
    lyapunov_exponent(sys, u0, dt; n_steps, n_renorm, epsilon) -> Float64

Вычисляет наибольший показатель Ляпунова методом перенормировки.

Запускает две траектории: основную из `u0` и возмущённую (`u0[1] += epsilon`).
Через каждые `n_renorm` шагов измеряет расстояние между ними, накапливает
`log(d / epsilon)`, затем сбрасывает возмущение обратно к `epsilon`, сохраняя
направление. Итоговый показатель: `λ = Σ log(d/ε) / (суммарное время)`.

Параметры:
- `n_steps`  — число шагов интегрирования (по умолчанию `10_000`)
- `n_renorm` — шагов между перенормировками (по умолчанию `10`)
- `epsilon`  — размер начального возмущения (по умолчанию `1e-8`)

Интерпретация результата:
- `λ > 0` — хаотическая система; горизонт предсказуемости ≈ `1/λ` единиц времени
- `λ ≈ 0` — граница хаоса
- `λ < 0` — устойчивые орбиты, регулярное поведение

# Пример
```julia
sys = Models.LorenzSystem(10.0, 28.0, 8/3)
λ = lyapunov_exponent(sys, [1.0, 1.0, 1.0], 0.01)  # ожидается ≈ 0.9
```
"""
function lyapunov_exponent(
    sys,
    u0       :: Vector{Float64},
    dt       :: Float64;
    n_steps  :: Int     = 10_000,
    n_renorm :: Int     = 10,
    epsilon  :: Float64 = 1e-8
) :: Float64

    dim = length(u0)

    # Основная траектория
    u = copy(u0)

    # Возмущённая траектория: сдвигаем первую координату на epsilon
    # (можно любую ненулевую компоненту — направление не критично)
    delta = zeros(dim)
    delta[1] = epsilon
    u_pert = u .+ delta

    lambda_sum     = 0.0    # накопленная сумма логарифмов растяжений
    n_renorm_done  = 0      # счётчик перенормировок

    for step in 1:n_steps

        # Один шаг RK4 для обеих траекторий
        u      = _rk4_step(sys, u,      dt)
        u_pert = _rk4_step(sys, u_pert, dt)

        # Перенормировка каждые n_renorm шагов
        if step % n_renorm == 0
            diff   = u_pert .- u              # вектор расхождения
            d_curr = _norm(diff)              # его длина

            # Защита от вырождения (если траектории совпали)
            if d_curr < 1e-15
                delta[1] = epsilon
                u_pert = u .+ delta
                continue
            end

            # Накапливаем логарифм коэффициента растяжения
            # log(d_curr / epsilon) = насколько выросло возмущение
            lambda_sum    += log(d_curr / epsilon)
            n_renorm_done += 1

            # Перенормируем: сохраняем направление, сбрасываем длину к epsilon
            u_pert = u .+ diff .* (epsilon / d_curr)
        end
    end

    # λ = (сумма логарифмов) / (суммарное время)
    # Делим на полное время = n_renorm_done * n_renorm * dt
    if n_renorm_done == 0
        return 0.0
    end

    total_time = n_renorm_done * n_renorm * dt
    return lambda_sum / total_time
end


# =============================================================================
# СТАТИСТИКА ТРАЕКТОРИИ
#
# Вычисляет набор числовых характеристик по уже готовой траектории.
# Принимает u_results из любого солвера.
#
# Содержимое TrajectoryStats:
#   mean_u       — среднее значение каждой координаты (центр аттрактора)
#   std_u        — среднеквадратичное отклонение (разброс)
#   min_u        — минимальные значения координат
#   max_u        — максимальные значения координат
#   mean_speed   — средняя скорость движения по фазовому пространству
#   max_speed    — максимальная скорость (в каком месте аттрактор "быстрый")
#   path_length  — длина траектории в фазовом пространстве
# =============================================================================

"""
    TrajectoryStats

Числовые характеристики траектории, вычисленные функцией `trajectory_stats`.

Поля:
- `mean_u`      — среднее значение каждой координаты (геометрический центр аттрактора)
- `std_u`       — стандартное отклонение по каждой координате
- `min_u`       — минимальные значения координат
- `max_u`       — максимальные значения координат
- `mean_speed`  — средняя скорость движения в фазовом пространстве (`‖Δu/Δt‖`)
- `max_speed`   — максимальная скорость (показывает, где аттрактор «разгоняется»)
- `path_length` — суммарная длина пути в фазовом пространстве
"""
struct TrajectoryStats
    mean_u      :: Vector{Float64}
    std_u       :: Vector{Float64}
    min_u       :: Vector{Float64}
    max_u       :: Vector{Float64}
    mean_speed  :: Float64
    max_speed   :: Float64
    path_length :: Float64
end

"""
    trajectory_stats(u_results, dt) -> TrajectoryStats

Вычисляет статистику по траектории `u_results`, полученной от любого солвера.
`dt` используется для расчёта скоростей из конечных разностей.
"""
function trajectory_stats(
    u_results :: Vector{Vector{Float64}},
    dt        :: Float64
) :: TrajectoryStats

    n   = length(u_results)
    dim = length(u_results[1])

    # --- Среднее по каждой координате ---
    mean_u = zeros(dim)
    for u in u_results
        mean_u .+= u
    end
    mean_u ./= n

    # --- Стандартное отклонение ---
    # std = sqrt( mean( (u - mean_u)² ) )
    std_u = zeros(dim)
    for u in u_results
        std_u .+= (u .- mean_u) .^ 2
    end
    std_u = sqrt.(std_u ./ n)

    # --- Минимумы и максимумы ---
    min_u = copy(u_results[1])
    max_u = copy(u_results[1])
    for u in u_results
        for d in 1:dim
            if u[d] < min_u[d]; min_u[d] = u[d]; end
            if u[d] > max_u[d]; max_u[d] = u[d]; end
        end
    end

    # --- Скорость и длина пути ---
    # Скорость в дискретном случае ≈ ||u[i+1] - u[i]|| / dt
    speeds      = Float64[]
    path_length = 0.0

    for i in 1:(n-1)
        step_dist = _norm(u_results[i+1] .- u_results[i])
        speed     = step_dist / dt
        push!(speeds, speed)
        path_length += step_dist
    end

    mean_speed = isempty(speeds) ? 0.0 : sum(speeds) / length(speeds)
    max_speed  = isempty(speeds) ? 0.0 : maximum(speeds)

    return TrajectoryStats(mean_u, std_u, min_u, max_u,
                           mean_speed, max_speed, path_length)
end


# =============================================================================
# СКОРОСТЬ РАСХОЖДЕНИЯ ТРАЕКТОРИЙ
#
# Более простая альтернатива показателю Ляпунова для визуализации.
# Вместо одного числа возвращает ВЕКТОР расстояний d(t) со временем.
#
# Это позволяет нарисовать график: как быстро расходятся траектории.
# На хаотической системе график будет экспоненциально растущим,
# на периодической — останется ограниченным.
#
# Возвращает:
#   t_steps    — моменты времени
#   distances  — расстояние между траекториями в каждый момент
#
# Перенормировки НЕТ — расстояние растёт до насыщения, это намеренно.
# Насыщение происходит когда траектории расходятся на размер аттрактора.
# =============================================================================

"""
    divergence_rate(sys, u0, dt; n_steps, epsilon) -> (t_steps, distances)

Отслеживает расстояние между двумя близкими траекториями без перенормировки.

Возвращает пару векторов: моменты времени и расстояния между траекториями.
На хаотической системе кривая расстояний растёт экспоненциально до насыщения
(когда траектории расходятся на размер аттрактора). На периодической системе
расстояние остаётся ограниченным. Интегрирование останавливается досрочно,
если расстояние превысило `1e6`.

В отличие от `lyapunov_exponent`, не возвращает одно число, а даёт полную
картину расхождения во времени — удобно для визуализации.

Параметры:
- `n_steps`  — максимальное число шагов (по умолчанию `5_000`)
- `epsilon`  — начальное возмущение (по умолчанию `1e-6`)
"""
function divergence_rate(
    sys,
    u0      :: Vector{Float64},
    dt      :: Float64;
    n_steps :: Int     = 5_000,
    epsilon :: Float64 = 1e-6
) :: Tuple{Vector{Float64}, Vector{Float64}}

    dim    = length(u0)
    delta  = zeros(dim)
    delta[1] = epsilon

    u      = copy(u0)
    u_pert = u .+ delta

    t_steps   = Float64[]
    distances = Float64[]

    push!(t_steps, 0.0)
    push!(distances, epsilon)

    for step in 1:n_steps
        u      = _rk4_step(sys, u,      dt)
        u_pert = _rk4_step(sys, u_pert, dt)

        d = _norm(u_pert .- u)
        push!(t_steps,   step * dt)
        push!(distances, d)

        # Останавливаемся если расстояние перестало расти (насыщение)
        if d > 1e6
            break
        end
    end

    return t_steps, distances
end


# =============================================================================
# ПЛОТНОСТЬ В ФАЗОВОМ ПРОСТРАНСТВЕ
#
# Разбивает фазовое пространство на сетку n×n ячеек и считает
# сколько точек траектории попало в каждую ячейку.
#
# Это показывает "где аттрактор проводит больше времени".
# Работает только для 2D — берёт первые две координаты u[1], u[2].
#
# Возвращает матрицу n×n: density[i,j] = число точек в ячейке (i,j).
# Нормировка: делим на общее число точек → получаем вероятность.
#
# Применение в GUI: можно нарисовать тепловую карту аттрактора.
# =============================================================================

"""
    phase_space_density(u_results; n_bins) -> (density, x_centers, y_centers)

Строит 2D-гистограмму плотности траектории по первым двум координатам.

Разбивает область значений `[u[1], u[2]]` на сетку `n_bins × n_bins` ячеек
и подсчитывает, сколько точек попало в каждую. Результат нормируется на единицу —
получается дискретная вероятностная плотность: «где аттрактор проводит больше времени».

Возвращает:
- `density`   — матрица `n_bins × n_bins` с вероятностями
- `x_centers` — центры ячеек по оси X
- `y_centers` — центры ячеек по оси Y

Параметры:
- `n_bins` — число ячеек по каждой оси (по умолчанию `50`)
"""
function phase_space_density(
    u_results :: Vector{Vector{Float64}};
    n_bins    :: Int = 50
) :: Tuple{Matrix{Float64}, Vector{Float64}, Vector{Float64}}

    # Собираем все значения первых двух координат
    xs = [u[1] for u in u_results]
    ys = [u[2] for u in u_results]

    x_min, x_max = minimum(xs), maximum(xs)
    y_min, y_max = minimum(ys), maximum(ys)

    # Небольшой отступ чтобы граничные точки не вышли за сетку
    margin = 0.01
    x_min -= margin * abs(x_min)
    x_max += margin * abs(x_max)
    y_min -= margin * abs(y_min)
    y_max += margin * abs(y_max)

    # Шаг сетки по каждой оси
    dx = (x_max - x_min) / n_bins
    dy = (y_max - y_min) / n_bins

    density = zeros(Float64, n_bins, n_bins)

    for (x, y) in zip(xs, ys)
        # Индекс ячейки: clamp защищает от выхода за границы из-за погрешностей
        i = clamp(floor(Int, (x - x_min) / dx) + 1, 1, n_bins)
        j = clamp(floor(Int, (y - y_min) / dy) + 1, 1, n_bins)
        density[i, j] += 1.0
    end

    # Нормировка → вероятностная плотность
    total = sum(density)
    if total > 0
        density ./= total
    end

    # Центры ячеек по каждой оси (для осей графика)
    x_centers = [x_min + (i - 0.5) * dx for i in 1:n_bins]
    y_centers = [y_min + (j - 0.5) * dy for j in 1:n_bins]

    return density, x_centers, y_centers
end


# =============================================================================
# ПРОВЕРКА СОХРАНЕНИЯ ИНТЕГРАЛА ДВИЖЕНИЯ
#
# Некоторые системы имеют сохраняющиеся величины (интегралы движения).
# Для Лотки-Вольтерры существует интеграл Ляпунова:
#
#   V(x, y) = δx - γ·ln(x) + βy - α·ln(y) = const
#
# Для двойного маятника сохраняется полная механическая энергия:
#
#   E = T + U  (кинетическая + потенциальная)
#
# Если численный метод точный — значение интеграла не должно меняться.
# Дрейф интеграла = мера накопленной ошибки солвера.
#
# Функция вычисляет интеграл в каждой точке траектории и возвращает:
#   values    — значения интеграла V(t)
#   drift     — относительный дрейф: (V_max - V_min) / |V_mean|
#               близко к 0 → солвер точный
#               велико     → солвер накапливает ошибку
# =============================================================================

"""
    conservation_check(sys::LotkaVolterraSystem, u_results) -> (values, drift)

Проверяет сохранение интеграла движения Лотки–Вольтерры:
`V = δx − γ ln x + βy − α ln y = const`.

Возвращает вектор значений `V` вдоль траектории и относительный дрейф
`(V_max − V_min) / |V_mean|`. Значения `NaN` появляются, если координаты
стали отрицательными (нефизичный режим).

Интерпретация дрейфа:
- `< 0.001` — солвер сохраняет интеграл отлично
- `0.001–0.01` — небольшая накопленная ошибка, можно уменьшить `dt`
- `> 0.01` — значительный дрейф, рекомендуется RK4 с меньшим шагом
"""
function conservation_check(
    sys       :: LotkaVolterraSystem,
    u_results :: Vector{Vector{Float64}}
) :: Tuple{Vector{Float64}, Float64}

    # Интеграл Ляпунова для Лотки-Вольтерры
    # V = δx - γ ln(x) + βy - α ln(y)
    values = Float64[]

    for u in u_results
        x, y = u[1], u[2]
        if x <= 0 || y <= 0
            push!(values, NaN)
            continue
        end
        V = sys.delta * x - sys.gamma * log(x) +
            sys.beta  * y - sys.alpha  * log(y)
        push!(values, V)
    end

    # Убираем NaN для вычисления статистики
    valid = filter(isfinite, values)
    if isempty(valid)
        return values, Inf
    end

    V_mean = sum(valid) / length(valid)
    V_min  = minimum(valid)
    V_max  = maximum(valid)

    drift = abs(V_mean) > 1e-10 ? (V_max - V_min) / abs(V_mean) : 0.0

    return values, drift
end

"""
    conservation_check(sys::DoublePendulumSystem, u_results) -> (values, drift)

Проверяет сохранение полной механической энергии двойного маятника:
`E = T + U`, где `T` — кинетическая, `U` — потенциальная энергия.

Возвращает вектор значений `E` вдоль траектории и относительный дрейф.
Дрейф энергии служит мерой накопленной численной ошибки.
"""
function conservation_check(
    sys       :: DoublePendulumSystem,
    u_results :: Vector{Vector{Float64}}
) :: Tuple{Vector{Float64}, Float64}

    # Полная механическая энергия двойного маятника
    # E = T + U
    # T = 0.5*(m1+m2)*L1²*ω1² + 0.5*m2*L2²*ω2² + m2*L1*L2*ω1*ω2*cos(θ1-θ2)
    # U = -(m1+m2)*g*L1*cos(θ1) - m2*g*L2*cos(θ2)

    m1, m2, L1, L2 = sys.m1, sys.m2, sys.L1, sys.L2
    g = 9.81

    values = Float64[]

    for u in u_results
        θ1, ω1, θ2, ω2 = u[1], u[2], u[3], u[4]

        T = 0.5*(m1+m2)*L1^2*ω1^2 +
            0.5*m2*L2^2*ω2^2 +
            m2*L1*L2*ω1*ω2*cos(θ1-θ2)

        U = -(m1+m2)*g*L1*cos(θ1) - m2*g*L2*cos(θ2)

        push!(values, T + U)
    end

    V_mean = sum(values) / length(values)
    V_min  = minimum(values)
    V_max  = maximum(values)

    drift = abs(V_mean) > 1e-10 ? (V_max - V_min) / abs(V_mean) : 0.0

    return values, drift
end


# =============================================================================
# СВОДНЫЙ ОТЧЁТ
#
# Красиво печатает все характеристики системы в консоль.
# Удобно вызвать один раз после интегрирования чтобы получить
# полную картину: хаос это или нет, насколько точен солвер, и т.д.
#
# Использование:
#   sys = Models.LorenzSystem(10.0, 28.0, 2.666)
#   _, u_res, _ = Solvers.solve_rk4(sys, [1.,1.,1.], (0.,50.), 0.01)
#   Analysis.print_analysis_report(sys, u_res, dt=0.01)
# =============================================================================

"""
    print_analysis_report(sys, u_results; dt)

Печатает в консоль полный анализ системы: размеры траектории, диапазоны и
статистику координат, показатель Ляпунова с интерпретацией, дрейф интеграла
движения (для `LotkaVolterraSystem` и `DoublePendulumSystem`).

# Пример
```julia
sys = Models.LorenzSystem(10.0, 28.0, 8/3)
_, u_res, _ = Solvers.solve_rk4(sys, [1.0, 1.0, 1.0], (0.0, 50.0), 0.01)
Analysis.print_analysis_report(sys, u_res; dt=0.01)
```
"""
function print_analysis_report(
    sys       :: Any,
    u_results :: Vector{Vector{Float64}};
    dt        :: Float64 = 0.01
)
    println("\n" * "="^60)
    println("  АНАЛИЗ СИСТЕМЫ: $(Models.system_name(sys))")
    println("="^60)

    # --- Статистика траектории ---
    stats = trajectory_stats(u_results, dt)
    n     = length(u_results)
    dim   = length(u_results[1])
    T     = (n - 1) * dt

    println("\n📐 ТРАЕКТОРИЯ")
    println("  Точек          : $n")
    println("  Время          : $T")
    println("  Длина пути     : $(round(stats.path_length, digits=2))")
    println("  Средняя скорость: $(round(stats.mean_speed, digits=4))")
    println("  Макс. скорость : $(round(stats.max_speed, digits=4))")

    println("\n📊 ДИАПАЗОНЫ КООРДИНАТ")
    coord_names = dim == 2 ? ["x", "y"] :
                  dim == 3 ? ["x", "y", "z"] :
                             ["θ1", "ω1", "θ2", "ω2"]
    for d in 1:dim
        name = coord_names[d]
        println("  $name: [$(round(stats.min_u[d], digits=3)), $(round(stats.max_u[d], digits=3))]" *
                "  среднее=$(round(stats.mean_u[d], digits=3))" *
                "  σ=$(round(stats.std_u[d], digits=3))")
    end

    # --- Показатель Ляпунова ---
    println("\n🌀 ПОКАЗАТЕЛЬ ЛЯПУНОВА")
    println("  Вычисляем... (5000 шагов)")
    λ = lyapunov_exponent(sys, u_results[1], dt, n_steps=5000)
    println("  λ = $(round(λ, digits=5))")
    if λ > 0.01
        println("  → ХАОТИЧЕСКАЯ система (λ > 0)")
        println("  → Горизонт предсказуемости ≈ $(round(1/λ, digits=2)) единиц времени")
    elseif λ > -0.01
        println("  → ГРАНИЦА ХАОСА (λ ≈ 0)")
    else
        println("  → РЕГУЛЯРНАЯ система (λ < 0), устойчивые орбиты")
    end

    # --- Проверка сохранения (если применимо) ---
    if sys isa LotkaVolterraSystem || sys isa DoublePendulumSystem
        println("\n⚖️  СОХРАНЕНИЕ ИНТЕГРАЛА ДВИЖЕНИЯ")
        _, drift = conservation_check(sys, u_results)
        println("  Относительный дрейф: $(round(drift * 100, digits=4))%")
        if drift < 0.001
            println("  → Отлично: солвер сохраняет интеграл")
        elseif drift < 0.01
            println("  → Хорошо: небольшой дрейф, шаг dt можно уменьшить")
        else
            println("  → Внимание: значительный дрейф, нужен меньший dt или RK4")
        end
    end

    println("\n" * "="^60 * "\n")
end


# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (внутренние, не экспортируются)
# =============================================================================

# Один шаг RK4 — используется внутри lyapunov_exponent и divergence_rate
# Дублирует логику Solvers.solve_rk4, но без накопления массива точек
# (нам нужно только текущее состояние, история не нужна)
function _rk4_step(sys, u::Vector{Float64}, dt::Float64) :: Vector{Float64}
    k1 = Models.get_derivative(sys, u)
    k2 = Models.get_derivative(sys, u .+ (dt/2) .* k1)
    k3 = Models.get_derivative(sys, u .+ (dt/2) .* k2)
    k4 = Models.get_derivative(sys, u .+ dt .* k3)
    return u .+ (dt/6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
end

# Евклидова норма вектора: sqrt(x₁² + x₂² + ... + xₙ²)
function _norm(v::Vector{Float64}) :: Float64
    return sqrt(sum(x^2 for x in v))
end

end # module Analysis