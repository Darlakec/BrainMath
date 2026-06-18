# Подключаем наш графический интерфейс
include("Gui.jl")
using .Gui

println("Запуск симуляции...")

# Julia 1.12: после include() функция появляется в новой "версии мира"
# invokelatest говорит использовать самую свежую версию функции
Base.invokelatest(Gui.run_app)

println("Окно успешно открыто.")