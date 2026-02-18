# Использование

Запуск примеров из соответствующих папок. Рекомендуется запускать команды из корня репозитория или перейти в нужную подпапку.

Парсинг / подготовка данных:

```powershell
cd src\parsing
python app.py
```

Регрессия (предсказание зарплат):

```powershell
cd src\regression
python app.py "path_to_data"
```

Классификация:

```powershell
cd src\classification
python app.py
```

Файлы с данными (`*.npy`, `predicted_salaries.csv`) находятся в соответствующих подпапках `src/*`.
