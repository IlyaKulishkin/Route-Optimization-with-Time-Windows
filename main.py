from flask import Flask, render_template, request, jsonify
import numpy as np
from datetime import datetime
import requests
import random
import math
import time

app = Flask(__name__)


class City:
    def __init__(self, index, open_time, close_time, service_time=0):
        self.index = index
        self.open_time = open_time
        self.close_time = close_time
        self.service_time = service_time


"""Получает расстояние и время в пути между двумя точками, используя OSRM API."""


def get_route_distance(start, end):
    start_str = f"{start[1]},{start[0]}"
    end_str = f"{end[1]},{end[0]}"
    url = f"https://router.project-osrm.org/route/v1/driving/{start_str};{end_str}?overview=false"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data["code"] == "Ok":
            distance = data["routes"][0]["distance"] / 1000  # Convert to km
            duration = data["routes"][0]["duration"] / 60  # Convert to minutes
            return distance, duration
        else:
            print(f"OSRM API returned non-Ok status: {data['code']}")
            return None, None
    except requests.exceptions.RequestException as e:
        print(f"Error occurred while making request to OSRM API: {e}")
        return None, None


""" Создает матрицу расстояний и длительностей поездок между всеми точками."""


def coordinates_to_distance_matrix(coordinates):
    num_cities = len(coordinates)
    distances = np.zeros((num_cities, num_cities))
    durations = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            distance, duration = get_route_distance(coordinates[i], coordinates[j])
            if distance is not None:
                distances[i, j] = distances[j, i] = distance
                durations[i, j] = durations[j, i] = duration
            else:
                # Fallback to direct distance if routing fails
                distances[i, j] = distances[j, i] = np.linalg.norm(np.array(coordinates[i]) - np.array(coordinates[j]))
                durations[i, j] = durations[j, i] = distances[i, j] / 50  # Assume 50 km/h average speed
    return distances, durations


"""Конвертирует строку времени в минуты."""


def time_to_minutes(time_str):
    hours, minutes = map(int, time_str.split(':'))
    return hours * 60 + minutes


"""Конвертирует минуты в строку времени."""


def minutes_to_time(minutes):
    hours, mins = divmod(minutes, 60)
    return f"{hours:02d}:{mins:02d}"


"""Находит город с ближайшим временем открытия к начальному времени."""


def find_start_city(cities, start_time):
    start_minutes = start_time.hour * 60 + start_time.minute
    min_diff = float('inf')
    start_city = 0
    for i, city in enumerate(cities):
        diff = abs(city.open_time - start_minutes)
        if diff < min_diff:
            min_diff = diff
            start_city = i
    return start_city


"""Реализация алгоритма ветвей и границ для решения задачи коммивояжера с временными окнами."""


def branch_and_bound(current_city, visited, current_cost, best_cost, best_route, times, best_times, distances, cities,
                     current_time, speed):
    if len(visited) == len(cities):
        current_cost += distances[current_city][visited[0]]
        if current_cost < best_cost[0]:
            best_cost[0] = current_cost
            best_route[:] = visited[:] + [visited[0]]
            best_times[:] = times[:] + [current_time + distances[current_city][visited[0]] / speed * 60]
        return

    for next_city in range(len(cities)):
        if next_city not in visited:
            travel_time = distances[current_city][next_city] / speed * 60
            arrival_time = current_time + travel_time

            if arrival_time < cities[next_city].open_time:
                arrival_time = cities[next_city].open_time

            # Всегда добавляем город, даже если окно нарушено
            times.append(arrival_time)
            visited.append(next_city)
            branch_and_bound(next_city, visited, current_cost + distances[current_city][next_city],
                             best_cost, best_route, times, best_times, distances, cities,
                             arrival_time + cities[next_city].service_time, speed)
            visited.pop()
            times.pop()


"""Проверяет, возможно ли пройти маршрут с заданными временными окнами."""


def check_time_windows_feasibility(route, cities, start_time, distances, speed):
    current_time = start_time
    feasible = True
    arrival_times = [current_time]  # Начальное время
    for i in range(len(route) - 1):
        from_city = route[i]
        to_city = route[i + 1]

        # Расчет времени в пути
        travel_time = distances[from_city][to_city] / speed * 60  # в минутах

        # Время прибытия в следующий город
        arrival_time = current_time + travel_time

        # Проверка временного окна
        if arrival_time < cities[to_city].open_time:
            # Можем подождать открытия
            arrival_time = cities[to_city].open_time

        # Проверка на закрытие временного окна
        if arrival_time > cities[to_city].close_time:
            feasible = False

        arrival_times.append(arrival_time)
        current_time = arrival_time + cities[to_city].service_time

    return feasible, arrival_times


"""Вычисляет стоимость маршрута."""


def calculate_route_cost(route, distances):
    cost = 0
    for i in range(len(route) - 1):
        cost += distances[route[i]][route[i + 1]]
    return cost


"""Проверка валидности сегментов маршрута относительно временных окон."""


def check_segments_validity(route, times_in_minutes, cities):
    segments_valid = []
    problem_points = []

    for i in range(len(route) - 1):
        to_city_idx = route[i + 1]
        arrival_time = times_in_minutes[i + 1]

        valid = True
        if arrival_time < cities[to_city_idx].open_time:
            valid = False
        if arrival_time > cities[to_city_idx].close_time:
            valid = False

        segments_valid.append(valid)
        if not valid:
            problem_points.append(to_city_idx)

    return segments_valid, problem_points


"""Реализация алгоритма имитации отжига для задачи коммивояжера с временными окнами."""


def simulated_annealing(start_city, distances, cities, start_time, speed, max_iterations=10000,
                        initial_temperature=1000.0, cooling_rate=0.995):
    n = len(cities)

    # Создаем начальное решение - путь из начальной точки ко всем остальным и обратно
    current_route = [start_city]
    unvisited = list(range(n))
    unvisited.remove(start_city)
    current_route.extend(unvisited)
    current_route.append(start_city)

    # Проверяем выполнимость начального решения
    feasible, times = check_time_windows_feasibility(current_route, cities, start_time, distances, speed)

    # Текущая стоимость
    current_cost = calculate_route_cost(current_route, distances)

    # Лучшее решение и его стоимость
    best_route = current_route[:]
    best_cost = current_cost
    best_times = times[:]

    # Температура
    temperature = initial_temperature

    # Основной цикл алгоритма
    for iteration in range(max_iterations):
        # Выбираем случайные индексы для обмена (кроме первого и последнего, так как это начальная точка)
        i, j = random.sample(range(1, len(current_route) - 1), 2)

        # Создаем новый маршрут путем обмена двух городов
        new_route = current_route[:]
        new_route[i], new_route[j] = new_route[j], new_route[i]

        # Проверяем выполнимость нового решения
        feasible, new_times = check_time_windows_feasibility(new_route, cities, start_time, distances, speed)

        # Вычисляем стоимость нового маршрута
        new_cost = calculate_route_cost(new_route, distances)

        # Разница в стоимости
        cost_diff = new_cost - current_cost

        # Решение о принятии нового решения
        if cost_diff < 0 or random.random() < math.exp(-cost_diff / temperature):
            current_route = new_route
            current_cost = new_cost
            times = new_times

            # Обновляем лучшее решение, если текущее лучше
            if current_cost < best_cost:
                best_route = current_route[:]
                best_cost = current_cost
                best_times = times[:]

        # Уменьшаем температуру
        temperature *= cooling_rate

        # Если температура слишком низкая, останавливаем алгоритм
        if temperature < 1e-5:
            break

    return best_route, best_times, best_cost


"""Рендерит главную страницу."""


@app.route('/')
def index():
    return render_template('index.html')


"""
Обрабатывает POST-запрос для решения задачи коммивояжера.
Принимает данные о координатах, временных окнах, скорости и начальном времени.
Возвращает рассчитанный маршрут, времена прибытия и общее расстояние.
"""


@app.route('/solve', methods=['POST'])
def solution():
    try:
        # Получаем входные данные
        coordinates = request.form.get('coordinates')
        coordinates = np.array(eval(coordinates))
        time_windows = eval(request.form.get('time_windows'))
        speed = float(request.form.get('speed'))  # км/ч
        start_time = datetime.strptime(request.form.get('start_time'), '%H:%M')

        # Получаем матрицы расстояний и времен
        distances, durations = coordinates_to_distance_matrix(coordinates)

        n = len(coordinates)
        # Преобразуем временные окна в минуты
        cities = [City(i, time_to_minutes(time_windows[i][0]), time_to_minutes(time_windows[i][1]), 0) for i in
                  range(n)]

        # Находим стартовый город
        start_city = find_start_city(cities, start_time)
        start_minutes = max(start_time.hour * 60 + start_time.minute, cities[start_city].open_time)

        # Выбираем алгоритм в зависимости от размера задачи
        if n <= 10:
            # Для небольших задач используем ветви и границы
            best_cost = [float('inf')]
            best_route = []
            best_times = []
            start_alg_time = time.time()
            branch_and_bound(start_city, [start_city], 0, best_cost, best_route, [start_minutes], best_times,
                             distances, cities, start_minutes, speed)
            elapsed_time = time.time() - start_alg_time
            print(f"[Branch and Bound] Время выполнения: {elapsed_time:.4f} секунд")
            cost = best_cost[0]
            route = best_route
            times_minutes = best_times
            algorithm_name = 'Branch and Bound'
        else:
            # Для больших задач используем имитацию отжига
            start_alg_time = time.time()
            route, times_minutes, cost = simulated_annealing(
                start_city, distances, cities, start_minutes, speed
            )
            elapsed_time = time.time() - start_alg_time
            print(f"[Simulated Annealing] Время выполнения: {elapsed_time:.4f} секунд")
            algorithm_name = 'Simulated Annealing'

        if not route:
            return jsonify(error="Невозможно построить маршрут с введенными временными окнами")

        # Проверяем валидность сегментов маршрута
        segments_valid, problem_points = check_segments_validity(route, times_minutes, cities)

        # Конвертируем времена из минут в строковый формат
        times = [minutes_to_time(int(t)) for t in times_minutes]

        # Формируем координаты маршрута
        route_coordinates = [coordinates[i].tolist() for i in route]

        return jsonify({
            'route': route,
            'times': times,
            'cost': cost,
            'route_coordinates': route_coordinates,
            'algorithm': algorithm_name,
            'segments_valid': segments_valid,
            'problem_points': problem_points
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify(error=f"Произошла ошибка: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True)


