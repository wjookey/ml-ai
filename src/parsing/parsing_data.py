import pandas as pd
import numpy as np
import warnings
import re
import logging
from typing import Tuple
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder

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
            source_column: Название столбца с информацией о поле и возрасте.
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
            source_column: Название столбца с информацией о поле и возрасте.
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
                    age = int(age_match.group(1))
                    # Фильтрация некорректных значений возраста
                    if 18 <= age <= 70:  # Реалистичный диапазон возраста для работы
                        df.at[i, "age"] = age
                        extracted_count += 1
                except ValueError:
                    self.logger.warning(
                        f"Не удалось преобразовать возраст: {age_match.group(1)}"
                    )
                    continue

        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        self.logger.info(f"Извлечено {extracted_count} значений возраста")
        self.logger.info(f"Средний возраст: {df['age'].mean():.1f}")
        return super().handle(df)


class SalaryExtractor(DataHandler):
    """Извлекает и конвертирует зарплату в российские рубли."""

    def __init__(self, source_column: str) -> None:
        """Инициализирует извлекатель зарплаты.

        Аргументы:
            source_column: Название столбца с информацией о зарплате.
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

        df["salary"] = pd.to_numeric(df["salary"], errors="coerce")
        return super().handle(df)


class PositionExtractor(DataHandler):
    """Определяет желаемую должность."""

    def __init__(self, source_column: str) -> None:
        """Инициализирует извлекатель желаемой должности.

        Аргументы:
            source_column: Название столбца с информацией о желаемой должности.
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
            source_column: Название столбца с информацией о последней должности.
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
            source_column: Название столбца с информацией о городе.
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
            source_column: Название столбца с информацией о занятости.
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
            source_column: Название столбца с информацией о графике работы.
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
            source_column: Название столбца с информацией об опыте работы.
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
                r"опыт.*?работы.*?(\d+)\s*(?:месяцев|мес|месяца?)",
                text,
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
            if total_months > 0 and total_months <= 600:  # Максимум 50 лет
                df.at[i, "experience"] = total_months
                extracted_count += 1

        self.logger.info(f"Извлечено {extracted_count} значений опыта работы")
        if extracted_count > 0:
            self.logger.info(f"Средний опыт: {df['experience'].mean():.1f} месяцев")

        df["experience"] = pd.to_numeric(df["experience"], errors="coerce")
        return super().handle(df)


class EducationExtractor(DataHandler):
    """Извлекает уровень образования."""

    def __init__(self, source_column: str) -> None:
        """Инициализирует извлекатель образования.

        Аргументы:
            source_column: Название столбца с информацией об образовании.
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


class FeatureEngineering(DataHandler):
    """Добавляет новые признаки для улучшения предсказательной способности модели."""

    def __init__(self):
        super().__init__()

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавляет новые признаки.

        Аргументы:
            df: DataFrame с исходными данными.

        Вернет:
            DataFrame с добавленными признаками.
        """
        self.logger.info("Добавление новых признаков")
        df = df.copy()

        # 1. Взаимодействие возраста и опыта
        if "age" in df.columns and "experience" in df.columns:
            df["age"] = pd.to_numeric(df["age"], errors="coerce")
            df["experience"] = pd.to_numeric(df["experience"], errors="coerce")

            # Опыт по отношению к возрасту (сколько % жизни в работе)
            df["experience_to_age_ratio"] = df["experience"] / (df["age"] * 12)
            # Фильтрация некорректных значений
            df["experience_to_age_ratio"] = df["experience_to_age_ratio"].clip(0, 1)

        # 2. Группировка городов по регионам/кластерам
        if "city" in df.columns:
            # Определение столичных городов
            capital_cities = ["Москва", "Санкт-Петербург"]
            df["is_capital"] = df["city"].isin(capital_cities).astype(int)

            # Определение городов-миллионников
            million_cities = [
                "Москва",
                "Санкт-Петербург",
                "Новосибирск",
                "Екатеринбург",
                "Казань",
                "Нижний Новгород",
                "Челябинск",
                "Самара",
                "Омск",
                "Ростов-на-Дону",
                "Уфа",
                "Красноярск",
                "Воронеж",
                "Пермь",
                "Волгоград",
            ]
            df["is_million_city"] = df["city"].isin(million_cities).astype(int)

        # 3. Простота должности (количество слов)
        if "position" in df.columns:
            df["position_word_count"] = df["position"].apply(
                lambda x: len(str(x).split()) if pd.notna(x) else 0
            )

        # 4. Индикатор соответствия желаемой и последней должности
        if "position" in df.columns and "last position" in df.columns:
            df["position_match"] = (df["position"] == df["last position"]).astype(int)

        # 5. Плотность занятости (сколько вариантов выбрано)
        if "full time" in df.columns:
            employment_cols = [
                col
                for col in df.columns
                if col
                in [
                    "full time",
                    "part time",
                    "project work",
                    "work placement",
                    "volunteering",
                ]
            ]
            if employment_cols:
                df["employment_density"] = df[employment_cols].sum(axis=1)

        # 6. Плотность графика
        if "full day" in df.columns:
            schedule_cols = [
                col
                for col in df.columns
                if col
                in [
                    "full day",
                    "shift schedule",
                    "flexible schedule",
                    "rotation based work",
                    "remote working",
                ]
            ]
            if schedule_cols:
                df["schedule_density"] = df[schedule_cols].sum(axis=1)

        self.logger.info("Добавлено несколько новых признаков")
        return super().handle(df)


class OutlierHandler(DataHandler):
    """Обрабатывает выбросы в числовых признаках."""

    def __init__(self, iqr_multiplier: float = 3.0):
        """Инициализирует обработчик выбросов.

        Аргументы:
            iqr_multiplier: Множитель для определения границ выбросов.
        """
        super().__init__()
        self.iqr_multiplier = iqr_multiplier

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Обрабатывает выбросы в числовых признаках методом обрезания.

        Аргументы:
            df: DataFrame с данными.

        Вернет:
            DataFrame с обработанными выбросами.
        """
        self.logger.info("Обработка выбросов методом обрезания (clipping)")
        df = df.copy()

        # Определяем числовые признаки
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in numeric_cols:
            if col != "salary" and df[col].notna().any():
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - self.iqr_multiplier * IQR
                upper_bound = Q3 + self.iqr_multiplier * IQR

                # Обрезаем выбросы
                df[col] = df[col].clip(lower_bound, upper_bound)
                clipped_count = (
                    (df[col] == lower_bound) | (df[col] == upper_bound)
                ).sum()
                if clipped_count > 0:
                    self.logger.debug(f"  {col}: обрезано {clipped_count} выбросов")

        return super().handle(df)


class MissingValueHandler(DataHandler):
    """Обрабатывает пропущенные значения."""

    def __init__(
        self,
        numeric_strategy: str = "median",
        categorical_strategy: str = "most_frequent",
    ):
        """Инициализирует обработчик пропущенных значений.

        Аргументы:
            numeric_strategy: Стратегия для числовых признаков.
            categorical_strategy: Стратегия для категориальных признаков.
        """
        super().__init__()
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Обрабатывает пропущенные значения.

        Аргументы:
            df: DataFrame с данными.

        Вернет:
            DataFrame без пропущенных значений.
        """
        self.logger.info("Обработка пропущенных значений")
        df = df.copy()

        # Анализ пропущенных значений перед обработкой
        missing_before = df.isnull().sum().sum()
        self.logger.info(f"Пропущенных значений перед обработкой: {missing_before}")

        # Разделяем на числовые и категориальные признаки
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        # Удаляем salary из числовых признаков для отдельной обработки
        if "salary" in numeric_cols:
            numeric_cols.remove("salary")

        # Обработка числовых признаков
        for col in numeric_cols:
            if df[col].isnull().any():
                if self.numeric_strategy == "median":
                    fill_value = df[col].median()
                elif self.numeric_strategy == "mean":
                    fill_value = df[col].mean()
                elif self.numeric_strategy == "mode":
                    fill_value = df[col].mode()[0] if not df[col].mode().empty else 0
                else:
                    fill_value = 0

                df[col] = df[col].fillna(fill_value)
                self.logger.debug(
                    f"  {col} (числовой): заполнено {df[col].isnull().sum()} пропусков"
                )

        # Обработка категориальных признаков
        for col in categorical_cols:
            if df[col].isnull().any():
                if self.categorical_strategy == "most_frequent":
                    fill_value = (
                        df[col].mode()[0] if not df[col].mode().empty else "Не указано"
                    )
                elif self.categorical_strategy == "constant":
                    fill_value = "Не указано"
                else:
                    fill_value = "Не указано"

                df[col] = df[col].fillna(fill_value)
                self.logger.debug(
                    f"  {col} (категориальный): заполнено {df[col].isnull().sum()} пропусков"
                )

        # Анализ после обработки
        missing_after = df.isnull().sum().sum()
        self.logger.info(f"Пропущенных значений после обработки: {missing_after}")
        self.logger.info(
            f"Обработано {missing_before - missing_after} пропущенных значений"
        )

        return super().handle(df)


class SalaryOutlierHandler(DataHandler):
    """Отдельный обработчик выбросов в зарплате."""

    def __init__(self, method: str = "clip", iqr_multiplier: float = 3.0):
        """Инициализирует обработчик выбросов зарплаты.

        Аргументы:
            method: Метод обработки ('clip', 'remove').
            iqr_multiplier: Множитель для определения границ выбросов.
        """
        super().__init__()
        self.method = method
        self.iqr_multiplier = iqr_multiplier

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Обрабатывает выбросы в зарплате.

        Аргументы:
            df: DataFrame с данными.

        Вернет:
            DataFrame с обработанной зарплатой.
        """
        self.logger.info(f"Обработка выбросов в зарплате методом {self.method}")
        df = df.copy()

        if "salary" in df.columns and df["salary"].notna().any():
            Q1 = df["salary"].quantile(0.25)
            Q3 = df["salary"].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = max(
                0, Q1 - self.iqr_multiplier * IQR
            )  # Зарплата не может быть отрицательной
            upper_bound = Q3 + self.iqr_multiplier * IQR

            if self.method == "clip":
                # Обрезаем выбросы
                df["salary"] = df["salary"].clip(lower_bound, upper_bound)
                clipped_count = (
                    (df["salary"] == lower_bound) | (df["salary"] == upper_bound)
                ).sum()
                if clipped_count > 0:
                    self.logger.info(f"  Зарплата: обрезано {clipped_count} выбросов")
                    self.logger.info(
                        f"  Границы: {lower_bound:.0f} - {upper_bound:.0f}"
                    )

            elif self.method == "remove":
                # Удаляем строки с выбросами
                initial_len = len(df)
                df = df[(df["salary"] >= lower_bound) & (df["salary"] <= upper_bound)]
                removed_count = initial_len - len(df)
                if removed_count > 0:
                    self.logger.info(
                        f"  Зарплата: удалено {removed_count} строк с выбросами"
                    )

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
            f"  Конечные размеры: {initial_rows} строк, {final_columns} столбцов"
        )
        self.logger.info(f"  Удалено столбцов: {removed_columns}")

        return super().handle(df)


class SklearnPreprocessor(DataHandler):
    """Применяет стандартные sklearn преобразования."""

    def __init__(self):
        """Инициализирует sklearn препроцессор."""
        super().__init__()

    def handle(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Подготавливает данные для sklearn.

        Аргументы:
            df: DataFrame с данными.

        Вернет:
            Кортеж (X, y) для обучения модели.
        """
        self.logger.info("Подготовка данных для sklearn")
        df = df.copy()

        # Проверяем наличие целевой переменной
        if "salary" not in df.columns:
            raise ValueError("Целевая переменная 'salary' не найдена в данных")

        # Выделяем целевую переменную
        y = df["salary"].values
        X_df = df.drop(columns=["salary"], errors="ignore")

        # Разделяем на числовые и категориальные признаки
        numeric_features = X_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X_df.select_dtypes(exclude=[np.number]).columns.tolist()

        self.logger.info(f"Числовых признаков: {len(numeric_features)}")
        self.logger.info(f"Категориальных признаков: {len(categorical_features)}")

        numeric_transformer = SkPipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_transformer = SkPipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="constant", fill_value="missing"),
                ),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                (
                    "cat",
                    categorical_transformer,
                    categorical_features,
                ),
            ]
        )

        # Применяем преобразования
        X_processed = preprocessor.fit_transform(X_df)

        self.logger.info(f"Размер X после обработки: {X_processed.shape}")
        self.logger.info(f"Размер y: {y.shape}")

        return X_processed, y


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
        y = df["salary"].values

        self.logger.info("Целевая переменная (salary):")
        self.logger.info(f"  Размер: {y.shape}")
        self.logger.info(f"  Среднее: {np.nanmean(y):.2f}")
        self.logger.info(f"  Медиана: {np.nanmedian(y):.2f}")
        self.logger.info(f"  Стандартное отклонение: {np.nanstd(y):.2f}")

        X_df = df.drop(columns=["salary"], errors="ignore")

        # Преобразуем категориальные признаки в числовые
        X_df = pd.get_dummies(X_df)

        X = X_df.values

        self.logger.info("Матрица признаков X:")
        self.logger.info(f"  Размер: {X.shape}")
        self.logger.info(f"  Количество признаков: {X.shape[1]}")

        return X, y


class DataProcessingPipeline:
    """Пайплайн обработки данных с использованием цепочки ответственности."""

    def __init__(self, use_sklearn: bool = False) -> None:
        """Инициализирует пайплайн обработки данных.

        Аргументы:
            use_sklearn: Использовать ли sklearn preprocessing (стандартизацию и one-hot).
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.use_sklearn = use_sklearn
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
        salary_extractor = SalaryExtractor("ЗП")
        position_extractor = PositionExtractor("Ищет работу на должность:")
        last_position_extractor = LastPositionExtractor("Последеняя/нынешняя должность")
        city_extractor = CityExtractor("Город")
        employment_extractor = EmploymentExtractor("Занятость")
        schedule_extractor = ScheduleExtractor("График")
        experience_extractor = ExperienceExtractor(
            "Опыт (двойное нажатие для полной версии)"
        )
        education_extractor = EducationExtractor("Образование и ВУЗ")
        feature_engineering = FeatureEngineering()
        outlier_handler = OutlierHandler(iqr_multiplier=3.0)
        salary_outlier_handler = SalaryOutlierHandler(method="clip", iqr_multiplier=3.0)
        missing_value_handler = MissingValueHandler()
        column_cleaner = ColumnCleaner()

        if self.use_sklearn:
            data_preparer = SklearnPreprocessor()
        else:
            data_preparer = DataPreparer()

        # Строим цепочку
        sex_extractor.set_next(age_extractor).set_next(salary_extractor).set_next(
            salary_outlier_handler
        ).set_next(position_extractor).set_next(last_position_extractor).set_next(
            city_extractor
        ).set_next(
            employment_extractor
        ).set_next(
            schedule_extractor
        ).set_next(
            experience_extractor
        ).set_next(
            education_extractor
        ).set_next(
            feature_engineering
        ).set_next(
            outlier_handler
        ).set_next(
            missing_value_handler
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
            df: Исходный DataFrame с данными.

        Вернет:
            Кортеж (X, y) где X - признаки, y - целевая переменная.
        """
        self.logger.info(f"Начало обработки данных, размер: {df.shape}")
        try:
            result = self._pipeline.handle(df)
            self.logger.info("Обработка данных завершена успешно")
            return result
        except Exception as e:
            self.logger.error(
                f"Ошибка при обработке данных: {str(e)}",
                exc_info=True,
            )
            raise

    def save(
        self,
        X: np.ndarray,
        y: np.ndarray,
        x_path: str = "X_data.npy",
        y_path: str = "y_data.npy",
    ) -> None:
        """Сохраняет обработанные данные в npy файлы.

        Аргументы:
            X: Признаки.
            y: Целевая переменная.
            x_path: Путь для сохранения файла с признаками.
            y_path: Путь для сохранения файла с целевой переменной.
        """
        self.logger.info("Сохранение обработанных данных в npy")
        self.logger.info(f"Пути сохранения: X -> {x_path}, y -> {y_path}")

        try:
            # Сохраняем данные
            np.save(x_path, X)
            np.save(y_path, y)

            self.logger.info("Данные успешно сохранены:")
            self.logger.info(f"  X shape: {X.shape}")
            self.logger.info(f"  y shape: {y.shape}")
            self.logger.info(f"  X сохранен в: {x_path}")
            self.logger.info(f"  y сохранен в: {y_path}")

        except Exception as e:
            self.logger.error(
                f"Ошибка при сохранении данных: {str(e)}",
                exc_info=True,
            )
            raise
