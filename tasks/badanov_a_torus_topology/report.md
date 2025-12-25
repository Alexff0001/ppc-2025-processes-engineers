# Топология Решетка-тор

- Студент: Баданов Александр, группа 3823Б1ПР2
- Технология: SEQ, MPI
- Вариант: 9

## 1. Введение
Топология решетка-тор (torus topology) представляет собой двумерную сетку узлов, где каждый узел соединен со своими соседями, а края сети соединены между собой, образуя тор. Данная топология широко используется в суперкомпьютерах и высокопроизводительных вычислениях благодаря своей регулярности и эффективности маршрутизации сообщений.

## 2. Постановка задачи
Реализовать алгоритм работы с тороидальной топологией, включая преобразование координат, расчет расстояний и маршрутизацию данных между узлами.

Входные данные:
- Номер узла-источника (src)
- Номер узла-назначения (dst)
- Вектор данных для передачи (std::vector<double>)

Выходные данные: Вектор данных, доставленный узлу-назначению

## 3. Базовый алгоритм (Последовательный)
```cpp
bool BadanovATorusTopologySEQ::RunImpl() {
  const auto &in = GetInput();
  const size_t src = std::get<0>(in);
  const size_t dst = std::get<1>(in);
  const auto &data = std::get<2>(in);
  
  const int grid_size = 10;
  const int virtual_size = grid_size * grid_size;
  
  int src_rank = static_cast<int>(src) % virtual_size;
  int dst_rank = static_cast<int>(dst) % virtual_size;
  
  TorusCoords src_coords = RankToCoords(src_rank, grid_size);
  TorusCoords dst_coords = RankToCoords(dst_rank, grid_size);
  
  double distance = CalculateTorusDistance(src_coords, dst_coords, grid_size);
  
  std::vector<double> result = data;
  
  double scale = 1.0 / (1.0 + distance);
  for (auto& val : result) {
    val *= scale;
  }
  
  GetOutput() = result;
  return true;
}
```
Алгоритм последовательной версии моделирует эффект расстояния в топологии Решетка-тор. Он преобразует номера узлов в координаты, рассчитывает кратчайшее расстояние между ними и масштабирует данные обратно пропорционально этому расстоянию.

## 4. Схема распараллеливания
### Распределение вычислений
В MPI версии каждый процесс представляет отдельный узел тороидальной сети. Процессы самостоятельно определяют свою роль в маршруте передачи данных на основе глобальных координат.

### Коммуникационная схема
1. Определение геометрии сети:
- Автоматический расчет размеров сетки на основе общего числа процессов
- Построение прямоугольной или квадратной топологии

2. Построение маршрута:
```cpp
int dx = dst_coords.x - src_coords.x;
int dy = dst_coords.y - src_coords.y;

if (dx > cols / 2) dx -= cols;
else if (dx < -cols / 2) dx += cols;

if (dy > rows / 2) dy -= rows;
else if (dy < -rows / 2) dy += rows;
```
3. Передача данных:

- Узел-источник: отправляет данные следующему узлу
- Промежуточные узлы: ретранслируют данные
- Узел-назначение: принимает финальные данные

## 5. Детали реализации
### Структура кода
- `ops_mpi.cpp` - MPI реализация
- `ops_seq.cpp` - SEQ реализация
- `common.hpp` - общие типы данных
- Тесты в папках `tests/functional/` и `tests/performance/`

### Особенности реализации
- Гибкая топология: автоматическая адаптация под любое количество процессов
- Оптимальные маршруты: использование кратчайших путей с учетом тороидальности
- Асинхронная коммуникация: эффективное использование MPI_Send/MPI_Recv
- Комплексная валидация: проверка корректности входных параметров и состояний

## 6. Экспериментальная установка
### Оборудование и ПО
- **Процессор:** Apple M1
- **ОС:** macOS 15.3.1
- **Компилятор:** clang version 21.1.5
- **Тип сборки:** release
- **MPI:** Open MPI v5.0.8

### Данные для тестирования
Функциональные тесты:
- 15 тестовых случаев с различными комбинациями параметров
- Разные размеры сообщений (от 1 до 10000 элементов)
- Различные шаблоны маршрутизации (соседи, противоположные узлы, отправка самому себе)

Производительные тесты:
- Small: 10,000 элементов, короткие маршруты
- Medium: 100,000 элементов, средние маршруты
- Large: 1,000,000 элементов, длинные маршруты
- Self: 500,000 элементов, отправка самому себе
- Neighbor: 500,000 элементов, отправка соседнему узлу

## 7. Результаты и обсуждение
### Проверка корректности
Корректность проверялась с помощью 15 функциональных тестов, включающих:

- Преобразование координат
- Расчет кратчайших расстояний
- Маршрутизацию данных между различными узлами сети
- Граничные случаи (отправка самому себе, работа с пустыми данными)

### Производительность

| Процессы | Время, с | Ускорение | Эффективность |
|----------|-----------|-----------|---------------|
| 1 (SEQ)  |    0,13   | 1.00      | N/A           |
| 2        |    0,19   | 0,68      | 34%           |
| 4        |    0,42   | 0,31      | 10%            |
| 8        |    1,06   | 0,20      | 3%            |


## 8. Выводы
В ходе работы была успешно реализована тороидальная топология с поддержкой как последовательных, так и параллельных вычислений.

### Ограничения
- В SEQ версии только моделирование без реальной передачи данных
- Накладные расходы MPI
- Снижение эффективности при большом количестве процессов


## 9. Источники
1. Курс лекций по параллельному программированию Сысоева Александра Владимировича. 
2. Документация по курсу: https://learning-process.github.io/parallel_programming_course/ru

## Приложение

```cpp
bool BadanovATorusTopologyMPI::RunImpl() {
  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const auto &in = GetInput();
  const size_t src = std::get<0>(in);
  const size_t dst = std::get<1>(in);
  const auto &data = std::get<2>(in);

  auto &out = GetOutput();
  out.clear();

  const int src_rank = static_cast<int>(src);
  const int dst_rank = static_cast<int>(dst);

  if (src_rank == dst_rank) {
    if (world_rank == src_rank) {
      out = data;
    }
    return true;
  }

  int rows = static_cast<int>(std::sqrt(world_size));
  while (rows > 0 && world_size % rows != 0) {
    rows--;
  }
  int cols = world_size / rows;

  std::vector<int> route = GetRoute(src_rank, dst_rank, rows, cols);

  bool is_in_route = false;
  int position_in_route = -1;

  for (size_t i = 0; i < route.size(); i++) {
    if (world_rank == route[i]) {
      is_in_route = true;
      position_in_route = static_cast<int>(i);
      break;
    }
  }

  if (!is_in_route) {
    return true;
  }

  const int tag_data = 0;

  if (position_in_route == 0) {
    if (route.size() > 1) {
      int next_hop = route[1];
      MPI_Send(data.data(), static_cast<int>(data.size()), MPI_DOUBLE, next_hop, tag_data, MPI_COMM_WORLD);
    }
  } else if (position_in_route == static_cast<int>(route.size()) - 1) {
    out.resize(data.size());
    MPI_Recv(out.data(), static_cast<int>(out.size()), MPI_DOUBLE, MPI_ANY_SOURCE, tag_data, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  } else {
    std::vector<double> buffer(data.size());
    MPI_Recv(buffer.data(), static_cast<int>(buffer.size()), MPI_DOUBLE, MPI_ANY_SOURCE, tag_data, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);

    int next_hop = route[position_in_route + 1];
    MPI_Send(buffer.data(), static_cast<int>(buffer.size()), MPI_DOUBLE, next_hop, tag_data, MPI_COMM_WORLD);
  }

  return true;
}
```