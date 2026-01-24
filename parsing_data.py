import pandas as pd
import numpy as np
import warnings
import re
import logging
from typing import Tuple
from sklearn.preprocessing import MultiLabelBinarizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("data_processing.log"),
        logging.StreamHandler(),
    ],
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")


class DataHandler:
    """Базовый абстрактный класс обработчика данных в паттерне
    "Цепочка ответственности".
    """

    def __init__(self) -> None:
        """Инициализирует обработчик данных."""
        self._next_handler = None
        self.logger = logging.getLogger(self.__class__.__name__)

    def set_next(self, handler: "DataHandler") -> "DataHandler":
        """Устанавливает следующий обработчик в цепочке.

        Аргументы:
            handler: Следующий обработчик в цепочке.

        Вернет:
            Установленный обработчик для цепочного вызова.
        """
        self._next_handler = handler
        return handler

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Обрабатывает DataFrame и передает следующему обработчику.

        Аргументы:
            df: Входной DataFrame для обработки.

        Вернет:
            Обработанный DataFrame.
        """
        if self._next_handler:
            return self._next_handler.handle(df)
        return df


class SexExtractor(DataHandler):
    """Извлекает пол кандидата из текстового поля."""

    def __init__(self, source_column: str) -> None:
        """Инициализирует извлекатель пола.

        Аргументы:
            source_column: Название столбца с информацией о поле и возрасте,
                           по умолчанию "Пол, возраст".
        """
        super().__init__()
        self.source_column = source_column
        self._sex_pattern = r"(мужчина|male|женщина|female)"

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Извлекает информацию о поле из текста.

        Аргументы:
            df: DataFrame с исходными данными.

        Вернет:
            DataFrame с добавленным столбцом 'sex'.
        """
        self.logger.info(f"Извлечение пола из столбца {self.source_column}")
        df = df.copy()
        df["sex"] = None
        extracted_count = 0

        for i, text in enumerate(df[self.source_column]):
            text = str(text)
            sex_match = re.search(self._sex_pattern, text, re.IGNORECASE)
            if sex_match:
                sex = sex_match.group(1).lower()
                if "муж" in sex or "male" in sex:
                    df.at[i, "sex"] = "M"
                    extracted_count += 1
                elif "жен" in sex or "female" in sex:
                    df.at[i, "sex"] = "Ж"
                    extracted_count += 1

        self.logger.info(f"Извлечено {extracted_count} значений пола")
        return super().handle(df)


class AgeExtractor(DataHandler):
    """Извлекает возраст из текстового поля."""

    def __init__(self, source_column: str) -> None:
        """Инициализирует извлекатель возраста.

        Аргументы:
            source_column: Название столбца с информацией о поле и возрасте,
                           по умолчанию "Пол, возраст".
        """
        super().__init__()
        self.source_column = source_column
        self._age_pattern = r"(\d+)\s+(?:лет|год|года|years?)"

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Извлекает возраст из текста с помощью регулярных выражений.

        Аргументы:
            df: DataFrame с исходными данными.

        Вернет:
            DataFrame с добавленным столбцом 'age'.
        """
        self.logger.info(f"Извлечение возраста из столбца {self.source_column}")
        df = df.copy()
        df["age"] = None
        extracted_count = 0

        for i, text in enumerate(df[self.source_column]):
            text = str(text)
            age_match = re.search(self._age_pattern, text, re.IGNORECASE)
            if age_match:
                try:
                    df.at[i, "age"] = int(age_match.group(1))
                    extracted_count += 1
                except ValueError:
                    self.logger.warning(
                        f"Не удалось преобразовать возраст: {age_match.group(1)}"
                    )
                    continue

        df["age"] = pd.to_numeric(df["age"], errors="coerce").astype("Int64")
        self.logger.info(f"Извлечено {extracted_count} значений возраста")
        return super().handle(df)


class IQRFilter(DataHandler):
    """Фильтрует выбросы по методу межквартильного размаха (IQR)."""

    def __init__(self, column_name: str, iqr_multiplier: float = 3.5) -> None:
        """Инициализирует фильтр возраста.

        Аргументы:
            column_name: Название столбца, по умолчанию "age".
            iqr_multiplier: Множитель для IQR при определении границ выбросов,
                            по умолчанию 3.5.
        """
        super().__init__()
        self.column_name = column_name
        self.iqr_multiplier = iqr_multiplier

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Удаляет выбросы в данных.

        Аргументы:
            df: DataFrame с данными.

        Вернет:
            DataFrame без выбросов.
        """
        self.logger.info(f"Фильтрация выбросов в столбце {self.column_name}")
        df = df.copy()
        initial_len = len(df)

        if self.column_name in df.columns and df[self.column_name].notna().any():
            Q1 = df[self.column_name].quantile(0.25)
            Q3 = df[self.column_name].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - self.iqr_multiplier * IQR
            upper_bound = Q3 + self.iqr_multiplier * IQR

            self.logger.debug(f"Q1={Q1}, Q3={Q3}, IQR={IQR}")
            self.logger.debug(f"Границы: нижняя={lower_bound}, верхняя={upper_bound}")

            filtered_df = df[
                (df[self.column_name] >= lower_bound)
                & (df[self.column_name] <= upper_bound)
            ]

            removed_count = initial_len - len(filtered_df)
            if removed_count > 0:
                self.logger.info(
                    f"Удалено {removed_count} выбросов (было {initial_len}, стало {len(filtered_df)})"
                )
            else:
                self.logger.info("Выбросы не обнаружены")

            df = filtered_df

        return super().handle(df)


class SalaryExtractor(DataHandler):
    """Извлекает и конвертирует зарплату в российские рубли."""

    def __init__(self, source_column: str) -> None:
        """Инициализирует извлекатель зарплаты.

        Аргументы:
            source_column: Название столбца с информацией о зарплате,
                           по умолчанию "ЗП".
        """
        super().__init__()
        self.source_column = source_column
        self._exchange_rates = {
            "RUB": 1,
            "USD": 63.9966,
            "EUR": 71.1194,
            "KZT": 0.1645,
            "UZS": 0.0068,
            "UAH": 2.5415,
            "BYN": 31.2927,
            "AZN": 37.7227,
            "KGS": 0.9147,
        }

        self._currency_patterns = {
            "RUB": r"(RUB|руб|rph|₽|RUR|РУБ)",
            "USD": r"(USD|\$|долл|usd)",
            "EUR": r"(EUR|€|евро|eur)",
            "KZT": r"(KZT|тенге|kzt)",
            "KGS": r"(KGS|сом|kgs)",
            "UAH": r"(UAH|грив|uah)",
            "BYN": r"(BYN|бел|byn)",
            "AZN": r"(AZN|манат|azn)",
            "UZS": r"(UZS|uzs|сум|som|cym)",
        }

        self._num_pattern = r"([\d\s]+)"

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Извлекает зарплату и конвертирует в рубли.

        Аргументы:
            df: DataFrame с исходными данными.

        Вернет:
            DataFrame с добавленным столбцом 'salary' в рублях.
        """
        self.logger.info(f"Извлечение зарплаты из столбца {self.source_column}")
        df = df.copy()
        df["salary"] = None
        extracted_count = 0
        conversion_stats = {currency: 0 for currency in self._exchange_rates}

        for i, text in enumerate(df[self.source_column]):
            text = str(text).strip()
            num_salary = 0

            # Извлечение числового значения
            cleaned_text = re.sub(r"[^0-9,.\-]", "", text).replace(",", ".")
            num_match = re.search(self._num_pattern, cleaned_text)

            if num_match:
                try:
                    num_salary = float(num_match.group(1).replace(" ", ""))
                except (ValueError, AttributeError):
                    self.logger.debug(
                        f"Не удалось преобразовать число в тексте: {text}"
                    )

            # Конвертация в рубли
            text = text.upper()
            for currency, pattern in self._currency_patterns.items():
                if re.search(pattern, text, re.IGNORECASE):
                    num_salary *= self._exchange_rates[currency]
                    conversion_stats[currency] += 1
                    break

            if num_salary > 0:
                df.at[i, "salary"] = num_salary
                extracted_count += 1

        self.logger.info(f"Извлечено {extracted_count} значений зарплаты")
        for currency, count in conversion_stats.items():
            if count > 0:
                self.logger.debug(f"Конвертация из {currency}: {count} раз")

        df["salary"] = pd.to_numeric(df["salary"], errors="coerce").astype("Float64")
        return super().handle(df)


class PositionExtractor(DataHandler):
    """Определяет желаемую должность."""

    def __init__(self, source_column: str) -> None:
        """Инициализирует извлекатель желаемой должности.

        Аргументы:
            source_column: Название столбца с информацией о желаемой должности,
                           по умолчанию "Ищет работу на должность:".
        """
        super().__init__()
        self.source_column = source_column
        self._positions = {
            "Системный администратор": r"системный администратор|сетевой администратор|"
            r"сисадмин|system administrator|network administrator|sysadmin",
            "Инженер": r"мастер|инженер|технический специалист|техник|"
            r"специалист|repairman|engineer|technical specialist|"
            r"technician|specialist",
            "Аналитик": r"аналитик|системный аналитик|аналитик данных|"
            r"бизнес-аналитик|analyst|system analyst|data analyst|"
            r"business analyst|data scientist",
            "Менеджер": r"менеджер|руководитель|директор|начальник|"
            r"manager|supervisor",
            "Специалист технической поддержки": r"поддержки|поддержке|техподдержки|техподдержка|"
            r"оператор|модератор|support|technical support|operator|moderator",
            "Программист": r"программист 1С|1С-программист|1С программист|"
            r"1С-разработчик|разработчик 1С|1С разработчик|"
            r"разработчик|программист|developer|software engineer",
            "Тестировщик": r"тестировщик|QA|quality assurance|devops|test",
            "Маркетолог": r"маркетолог|реклама|marketing|smm|seo",
            "Дизайнер": r"дизайнер|верстальщик|иллюстратор|художник|UI|UX|"
            r"design|designer|illustrator|2d|3d|artist",
            "Монтажник": r"монтажник",
        }

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Определяет желаемую должность по ключевым словам.

        Аргументы:
            df: DataFrame с исходными данными.

        Вернет:
            DataFrame с добавленным столбцом 'position'.
        """
        self.logger.info(f"Извлечение должности из столбца {self.source_column}")
        df = df.copy()
        df["position"] = None
        position_stats = {position: 0 for position in self._positions.keys()}
        position_stats["Другая"] = 0

        for i, text in enumerate(df[self.source_column]):
            text = str(text).strip()
            position_found = False
            for position, pattern in self._positions.items():
                if re.search(pattern, text, re.IGNORECASE):
                    df.at[i, "position"] = position
                    position_stats[position] += 1
                    position_found = True
                    break

            if not position_found:
                df.at[i, "position"] = "Другая"
                position_stats["Другая"] += 1

        self.logger.info("Статистика извлечения должностей:")
        for position, count in position_stats.items():
            if count > 0:
                self.logger.info(f"  {position}: {count}")

        return super().handle(df)


class LastPositionExtractor(DataHandler):
    """Извлекает последнюю/нынешнюю должность из истории работы."""

    def __init__(self, source_column: str) -> None:
        """Инициализирует извлекатель последней должности.

        Аргументы:
            source_column: Название столбца с информацией о последней должности,
                           по умолчанию "Последеняя/нынешняя должность".
        """
        super().__init__()
        self.source_column = source_column
        self._positions = {
            "Системный администратор": r"системный администратор|сетевой администратор|"
            r"сисадмин|system administrator|network administrator|sysadmin",
            "Инженер": r"мастер|инженер|технический специалист|техник|"
            r"специалист|repairman|engineer|technical specialist|"
            r"technician|specialist",
            "Аналитик": r"аналитик|системный аналитик|аналитик данных|"
            r"бизнес-аналитик|analyst|system analyst|data analyst|"
            r"business analyst|data scientist",
            "Менеджер": r"менеджер|руководитель|директор|начальник|"
            r"manager|supervisor",
            "Специалист технической поддержки": r"поддержки|поддержке|техподдержки|техподдержка|"
            r"оператор|модератор|support|technical support|"
            r"operator|moderator",
            "Программист": r"программист 1С|1С-программист|1С программист|"
            r"1С-разработчик|разработчик 1С|1С разработчик|"
            r"разработчик|программист|developer|software engineer",
            "Тестировщик": r"тестировщик|QA|quality assurance|devops|test",
            "Маркетолог": r"маркетолог|реклама|marketing|smm|seo",
            "Дизайнер": r"дизайнер|верстальщик|иллюстратор|художник|UI|UX|"
            r"design|designer|illustrator|2d|3d|artist",
            "Монтажник": r"монтажник",
        }

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Определяет последнюю/нынешнюю должность по ключевым словам.

        Аргументы:
            df: DataFrame с исходными данными.

        Вернет:
            DataFrame с добавленным столбцом 'last position'.
        """
        df = df.copy()
        df["last position"] = None
        position_stats = {position: 0 for position in self._positions.keys()}
        position_stats["Другая"] = 0

        for i, text in enumerate(df[self.source_column]):
            text = str(text).strip()
            position_found = False
            for position, pattern in self._positions.items():
                if re.search(pattern, text, re.IGNORECASE):
                    df.at[i, "last position"] = position
                    position_stats[position] += 1
                    position_found = True
                    break

            if not position_found:
                df.at[i, "last position"] = "Другая"
                position_stats["Другая"] += 1

        self.logger.info("Статистика извлечения должностей:")
        for position, count in position_stats.items():
            if count > 0:
                self.logger.info(f"  {position}: {count}")

        return super().handle(df)


class CityExtractor(DataHandler):
    """Извлекает город проживания и информацию о готовности
    к переезду и командировкам.
    """

    def __init__(self, source_column: str) -> None:
        """Инициализирует извлекатель информации о городе.

        Аргументы:
            source_column: Название столбца с информацией о городе,
                           по умолчанию "Город".
        """
        super().__init__()
        self.source_column = source_column

        self._cities = {
            "Москва": r"москва|moscow",
            "Санкт-Петербург": r"санкт-петербург|санкт петербург|"
            r"saint-petersburg|saint petersburg",
            "Новосибирск": r"новосибирск|novosibirsk",
            "Екатеринбург": r"екатеринбург|ekaterinburg",
            "Казань": r"казань|kazan",
            "Красноярск": r"краснодрск|krasnoyarsk",
            "Нижний Новгород": r"нижний новгород|nizhniy novgorod",
            "Челябинск": r"челябинск|chelyabinsk",
            "Уфа": r"уфа|ufa",
            "Краснодар": r"краснодар|krasnodar",
            "Самара": r"самара|samara",
            "Ростов-на-Дону": r"ростов-на-дону|ростов на дону|"
            r"rostov-on-don|rostov na donu",
            "Омск": r"омск|omsk",
            "Воронеж": r"воронеж|voronezh",
            "Пермь": r"пермь|perm",
            "Волгоград": r"волгоград|volgograd",
            "Саратов": r"саратов|saratov",
            "Тюмень": r"тюмень|tyumen",
        }

        self._relocation_patterns = {
            "not_ready": r"(?:^|,\s*)не готов к переезду|не готова к переезду|"
            r"not willing to relocate|not prepared to relocate",
            "ready": r"(?:^|,\s*)готов к переезду|готова к переезду|"
            r"willing to relocate|хочу переехать|want to relocate|"
            r"prepared to relocate\s*\(.*\)",
        }

        self._business_trip_patterns = {
            "not_ready": r"(?:^|,\s*)не готов к командировкам|не готова к командировкам|"
            r"not prepared for business trips",
            "ready": r"(?:^|,\s*)готов к командировкам|готова к командировкам|"
            r"prepared for business trips|готов к редким командировкам|готова к редким командировкам|"
            r"prepared for occasional business trips\s*\(.*\)",
        }

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Извлекает город и информацию о переезде/командировках.

        Аргументы:
            df: DataFrame с исходными данными.

        Вернет:
            DataFrame с добавленными столбцами 'city', 'relocation',
            и 'business trips'.
        """
        self.logger.info(
            f"Извлечение информации о городе из столбца {self.source_column}"
        )
        df = df.copy()
        df["city"] = None
        df["relocation"] = None
        df["business trips"] = None

        city_stats = {city: 0 for city in self._cities.keys()}
        city_stats["Другой город"] = 0
        relocation_stats = {"да": 0, "нет": 0, "не указано": 0}
        business_trip_stats = {"да": 0, "нет": 0, "не указано": 0}

        for i, text in enumerate(df[self.source_column]):
            text = str(text).strip()
            text_lower = text.lower()

            # Извлечение города
            found_city = False
            for city, pattern in self._cities.items():
                if re.search(pattern, text_lower):
                    df.at[i, "city"] = city
                    city_stats[city] += 1
                    found_city = True
                    break

            if not found_city:
                df.at[i, "city"] = "Другой город"
                city_stats["Другой город"] += 1

            # Извлечение информации о переезде
            if re.search(self._relocation_patterns["not_ready"], text_lower):
                df.at[i, "relocation"] = "нет"
                relocation_stats["нет"] += 1
            elif re.search(self._relocation_patterns["ready"], text_lower):
                df.at[i, "relocation"] = "да"
                relocation_stats["да"] += 1
            else:
                relocation_stats["не указано"] += 1

            # Извлечение информации о командировках
            if re.search(self._business_trip_patterns["not_ready"], text_lower):
                df.at[i, "business trips"] = "нет"
                business_trip_stats["нет"] += 1
            elif re.search(self._business_trip_patterns["ready"], text_lower):
                df.at[i, "business trips"] = "да"
                business_trip_stats["да"] += 1
            else:
                business_trip_stats["не указано"] += 1

        self.logger.info("Статистика по городам (топ-10):")
        for city, count in sorted(city_stats.items(), key=lambda x: x[1], reverse=True)[
            :10
        ]:
            if count > 0:
                self.logger.info(f"  {city}: {count}")

        self.logger.info(f"Переезд: {relocation_stats}")
        self.logger.info(f"Командировки: {business_trip_stats}")

        return super().handle(df)


class EmploymentExtractor(DataHandler):
    """Обрабатывает информацию о желаемой занятости."""

    def __init__(self, source_column: str) -> None:
        """Инициализирует извлекатель информации о занятости.

        Аргументы:
            source_column: Название столбца с информацией о занятости,
                           по умолчанию "Занятость".
        """
        super().__init__()
        self.source_column = source_column

        self._replacements = {
            "стажировка": "work placement",
            "волонтерство": "volunteering",
            "частичная занятость": "part time",
            "проектная работа": "project work",
            "полная занятость": "full time",
        }

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Преобразует информацию о занятости в one-hot encoding.

        Аргументы:
            df: DataFrame с исходными данными.

        Вернет:
            DataFrame с one-hot encoded столбцами для типов занятости.
        """
        self.logger.info(
            f"Извлечение информации о занятости из столбца {self.source_column}"
        )
        df = df.copy()

        # Применение замен
        replacement_count = 0
        for i, text in enumerate(df[self.source_column]):
            text = str(text)
            original_text = text
            for ru, en in self._replacements.items():
                text = text.replace(ru, en)
            if text != original_text:
                replacement_count += 1
            df.at[i, self.source_column] = text

        self.logger.debug(f"Выполнено {replacement_count} замен")

        # Приведение к нижнему регистру и разделение
        df[self.source_column] = df[self.source_column].str.strip().str.lower()
        df["employment_list"] = df[self.source_column].str.split(", ")

        # One-hot encoding
        mlb = MultiLabelBinarizer()
        encoded = mlb.fit_transform(df["employment_list"])

        # Добавление закодированных столбцов
        for i, col_name in enumerate(mlb.classes_):
            df[col_name] = encoded[:, i]

        self.logger.info(
            f"Создано {len(mlb.classes_)} бинарных столбцов занятости: {list(mlb.classes_)}"
        )

        return super().handle(df)


class ScheduleExtractor(DataHandler):
    """Обрабатывает информацию о желаемом графике работы."""

    def __init__(self, source_column: str) -> None:
        """Инициализирует извлекатель информации о графике работы.

        Аргументы:
            source_column: Название столбца с информацией о графике работы,
                           по умолчанию "График".
        """
        super().__init__()
        self.source_column = source_column

        self._replacements = {
            "гибкий график": "flexible schedule",
            "полный день": "full day",
            "сменный график": "shift schedule",
            "вахтовый метод": "rotation based work",
            "удаленная работа": "remote working",
        }

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Преобразует информацию о графике работы в one-hot encoding.

        Аргументы:
            df: DataFrame с исходными данными.

        Вернет:
            DataFrame с one-hot encoded столбцами для типов графика работы.
        """
        self.logger.info(
            f"Извлечение информации о графике работы из столбца {self.source_column}"
        )
        df = df.copy()

        # Применение замен
        replacement_count = 0
        for i, text in enumerate(df[self.source_column]):
            text = str(text)
            original_text = text
            for ru, en in self._replacements.items():
                text = text.replace(ru, en)
            if text != original_text:
                replacement_count += 1
            df.at[i, self.source_column] = text

        self.logger.debug(f"Выполнено {replacement_count} замен")

        # Приведение к нижнему регистру и разделение
        df[self.source_column] = df[self.source_column].str.strip().str.lower()
        df["schedule_list"] = df[self.source_column].str.split(", ")

        # One-hot encoding
        mlb = MultiLabelBinarizer()
        encoded = mlb.fit_transform(df["schedule_list"])

        # Добавление закодированных столбцов
        for i, col_name in enumerate(mlb.classes_):
            df[col_name] = encoded[:, i]

        self.logger.info(
            f"Создано {len(mlb.classes_)} бинарных столбцов графика: {list(mlb.classes_)}"
        )

        return super().handle(df)


class ExperienceExtractor(DataHandler):
    """Извлекает опыт работы в месяцах."""

    def __init__(self, source_column: str) -> None:
        """Инициализирует извлекатель опыта работы.

        Аргументы:
            source_column: Название столбца с информацией об опыте работы,
                           по умолчанию "Опыт (двойное нажатие для полной версии)".
        """
        super().__init__()
        self.source_column = source_column

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Извлекает общий опыт работы в месяцах.

        Аргументы:
            df: DataFrame с исходными данными.

        Вернет:
            DataFrame с добавленным столбцом 'experience' (в месяцах).
        """
        self.logger.info(f"Извлечение опыта работы из столбца {self.source_column}")
        df = df.copy()
        df["experience"] = 0
        extracted_count = 0

        for i, text in enumerate(df[self.source_column]):
            text = str(text).lower()
            years = 0
            months = 0

            # Поиск лет и месяцев в полном формате
            years_match = re.search(r"опыт.*?работы.*?(\d+)\s*(?:лет|год[ау]?)", text)
            if years_match:
                try:
                    years = int(years_match.group(1))
                except ValueError:
                    self.logger.debug(
                        f"Не удалось преобразовать годы: {years_match.group(1)}"
                    )

            months_match = re.search(
                r"опыт.*?работы.*?(\d+)\s*(?:месяцев|мес|месяца?)", text
            )
            if months_match:
                try:
                    months = int(months_match.group(1))
                except ValueError:
                    self.logger.debug(
                        f"Не удалось преобразовать месяцы: {months_match.group(1)}"
                    )

            # Поиск в коротком формате "N г. M м."
            if months == 0:
                short_match = re.search(r"опыт.*?(\d+)\s*[гг].*?(\d+)\s*[мм]", text)
                if short_match:
                    try:
                        years = int(short_match.group(1))
                        months = int(short_match.group(2))
                    except (ValueError, IndexError):
                        self.logger.debug(
                            f"Не удалось распознать короткий формат: {text}"
                        )

            total_months = years * 12 + months
            if total_months > 0:
                df.at[i, "experience"] = total_months
                extracted_count += 1

        self.logger.info(f"Извлечено {extracted_count} значений опыта работы")
        if extracted_count > 0:
            self.logger.info(f"Средний опыт: {df['experience'].mean():.1f} месяцев")

        df["experience"] = pd.to_numeric(df["experience"], errors="coerce").astype(
            "Int64"
        )
        return super().handle(df)


class EducationExtractor(DataHandler):
    """Извлекает уровень образования."""

    def __init__(self, source_column: str) -> None:
        """Инициализирует извлекатель образования.

        Аргументы:
            source_column: Название столбца с информацией об образовании,
                           по умолчанию "Образование и ВУЗ".
        """
        super().__init__()
        self.source_column = source_column

        self._education_levels = {
            "Высшее": r"высшее|higher education",
            "Другое": r"неоконченное высшее|среднее специальное|колледж|"
            r"среднее образование|special|college|incomplete|secondary education",
        }

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Определяет уровень образования кандидата.

        Аргументы:
            df: DataFrame с исходными данными.

        Вернет:
            DataFrame с добавленным столбцом 'education'.
        """
        self.logger.info(
            f"Извлечение уровня образования из столбца {self.source_column}"
        )
        df = df.copy()
        df["education"] = None
        education_stats = {"Высшее": 0, "Другое": 0, "Не указано": 0}

        for i, text in enumerate(df[self.source_column]):
            text = str(text).lower().strip()

            # Проверка "Другое" в первую очередь, так как оно более специфично
            if re.search(self._education_levels["Другое"], text):
                df.at[i, "education"] = "Другое"
                education_stats["Другое"] += 1
            elif re.search(self._education_levels["Высшее"], text):
                df.at[i, "education"] = "Высшее"
                education_stats["Высшее"] += 1
            else:
                education_stats["Не указано"] += 1

        self.logger.info(f"Статистика образования: {education_stats}")
        return super().handle(df)


class ColumnCleaner(DataHandler):
    """Удаляет ненужные столбцы и строки и подготавливает данные для сохранения."""

    def __init__(self) -> None:
        """Инициализирует очиститель столбцов."""
        super().__init__()
        self._columns_to_drop = [
            "Пол, возраст",
            "ЗП",
            "Ищет работу на должность:",
            "Город",
            "Занятость",
            "График",
            "Опыт (двойное нажатие для полной версии)",
            "Последенее/нынешнее место работы",
            "Последеняя/нынешняя должность",
            "Образование и ВУЗ",
            "Обновление резюме",
            "Авто",
            "employment_list",
            "schedule_list",
        ]

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Удаляет исходные столбцы после извлечения нужной информации.

        Аргументы:
            df: DataFrame с извлеченными данными.

        Вернет:
            DataFrame без исходных столбцов.
        """
        self.logger.info("Очистка столбцов")
        df = df.copy()

        initial_rows = len(df)
        initial_columns = len(df.columns)

        # Удаление строк с пропущенными значениями
        df = df.dropna(axis=0)
        rows_after_dropna = len(df)
        dropped_rows = initial_rows - rows_after_dropna
        if dropped_rows > 0:
            self.logger.warning(
                f"Удалено {dropped_rows} строк с пропущенными значениями"
            )

        # Удаляем только те столбцы, которые существуют в DataFrame
        existing_columns = [col for col in self._columns_to_drop if col in df.columns]
        df = df.drop(columns=existing_columns, axis=1)

        final_columns = len(df.columns)
        removed_columns = initial_columns - final_columns

        self.logger.info("Очистка завершена:")
        self.logger.info(
            f"  Исходные размеры: {initial_rows} строк, {initial_columns} столбцов"
        )
        self.logger.info(
            f"  Конечные размеры: {rows_after_dropna} строк, {final_columns} столбцов"
        )
        self.logger.info(f"  Удалено столбцов: {removed_columns}")

        return super().handle(df)


class DataPreparer(DataHandler):
    """Подготавливает данные для сохранения в npy файлы."""

    def handle(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Преобразует DataFrame в numpy массивы для машинного обучения.

        Аргументы:
            df: Обработанный DataFrame.

        Вернет:
            Кортеж (X, y) где X - признаки, y - целевая переменная (salary).

        Исключения:
            ValueError: Если столбец 'salary' не найден в данных.
        """
        self.logger.info("Подготовка данных для машинного обучения")
        df = df.copy()

        # Убеждаемся, что salary существует
        if "salary" not in df.columns:
            error_msg = "Столбец 'salary' не найден в данных"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # Разделяем на признаки и целевую переменную
        y = df["salary"]

        self.logger.info("Целевая переменная (salary):")
        self.logger.info(f"  Размер: {y.shape}")
        self.logger.info(f"  Среднее: {np.nanmean(y):.2f}")
        self.logger.info(f"  Медиана: {np.nanmedian(y):.2f}")
        self.logger.info(f"  Стандартное отклонение: {np.nanstd(y):.2f}")

        # Удаляем salary из признаков
        X_df = df.drop(columns=["salary"], errors="ignore")

        # Преобразуем категориальные признаки в числовые
        X_df = pd.get_dummies(X_df)

        # Конвертируем в numpy array
        X = X_df

        self.logger.info("Матрица признаков X:")
        self.logger.info(f"  Размер: {X.shape}")
        self.logger.info(f"  Количество признаков: {X.shape[1]}")
        self.logger.info(f"  Тип данных: {X.dtypes}")

        return X, y


class DataProcessingPipeline:
    """Пайплайн обработки данных с использованием цепочки ответственности."""

    def __init__(self) -> None:
        """Инициализирует пайплайн обработки данных."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self._pipeline = self._build_pipeline()

    def _build_pipeline(self) -> DataHandler:
        """Строит цепочку обработчиков.

        Вернет:
            Первый обработчик в цепочке.
        """
        self.logger.info("Построение пайплайна обработки данных")

        # Создаем обработчики
        sex_extractor = SexExtractor("Пол, возраст")
        age_extractor = AgeExtractor("Пол, возраст")
        age_filter = IQRFilter("age")
        salary_extractor = SalaryExtractor("ЗП")
        position_extractor = PositionExtractor("Ищет работу на должность:")
        last_position_extractor = LastPositionExtractor("Последеняя/нынешняя должность")
        city_extractor = CityExtractor("Город")
        employment_extractor = EmploymentExtractor("Занятость")
        schedule_extractor = ScheduleExtractor("График")
        experience_extractor = ExperienceExtractor(
            "Опыт (двойное нажатие для полной версии)"
        )
        experience_filter = IQRFilter("experience")
        education_extractor = EducationExtractor("Образование и ВУЗ")
        column_cleaner = ColumnCleaner()
        data_preparer = DataPreparer()

        # Строим цепочку
        sex_extractor.set_next(age_extractor).set_next(age_filter).set_next(
            salary_extractor
        ).set_next(position_extractor).set_next(last_position_extractor).set_next(
            city_extractor
        ).set_next(
            employment_extractor
        ).set_next(
            schedule_extractor
        ).set_next(
            experience_extractor
        ).set_next(
            experience_filter
        ).set_next(
            education_extractor
        ).set_next(
            column_cleaner
        ).set_next(
            data_preparer
        )

        self.logger.info("Пайплайн построен успешно")
        return sex_extractor

    def process(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Обрабатывает DataFrame и возвращает X и y данные.

        Аргументы:
            df: Исходный DataFrame с данными из hh.csv.

        Вернет:
            Кортеж (X, y) где X - признаки, y - целевая переменная.
        """
        self.logger.info(f"Начало обработки данных, размер: {df.shape}")
        try:
            result = self._pipeline.handle(df)
            self.logger.info("Обработка данных завершена успешно")
            return result
        except Exception as e:
            self.logger.error(f"Ошибка при обработке данных: {str(e)}", exc_info=True)
            raise

    def save(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        x_path: str = "X_data.npy",
        y_path: str = "y_data.npy",
    ) -> None:
        """Обрабатывает DataFrame и сохраняет результаты в npy файлы.

        Аргументы:
            df: Исходный DataFrame с данными из hh.csv.
            x_path: Путь для сохранения файла с признаками.
            y_path: Путь для сохранения файла с целевой переменной.
        """
        self.logger.info("Начало полной обработки и сохранения данных")
        self.logger.info(f"Пути сохранения: X -> {x_path}, y -> {y_path}")

        try:
            # Сохраняем данные
            np.save(x_path, X)
            np.save(y_path, y)

            self.logger.info("Данные успешно сохранены:")
            self.logger.info(f"  X_data shape: {X.shape}")
            self.logger.info(f"  y_data shape: {y.shape}")
            self.logger.info(f"  X сохранен в: {x_path}")
            self.logger.info(f"  y сохранен в: {y_path}")

        except Exception as e:
            self.logger.error(f"Ошибка при сохранении данных: {str(e)}", exc_info=True)
            raise
