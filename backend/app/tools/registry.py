"""Сборка инструментов для конкретного пользователя.

Ключевой приём: user_id замыкается внутри функций и НЕ попадает
в схему, которую видит модель. Модель физически не может запросить
чужие данные — у неё нет такого параметра.
"""

from collections.abc import Iterable

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.tools import calendar as calendar_tools
from app.tools import nutrition as nutrition_tools
from app.tools import profile as profile_tools
from app.tools import recovery as recovery_tools
from app.tools import workout as workout_tools

ToolDomain = str
READ_ONLY_TOOL_NAMES = frozenset(
    {
        "get_my_profile",
        "search_food",
        "get_daily_intake",
        "get_workout_history",
        "get_recovery_logs",
        "get_weight_trend",
        "get_cycle_logs",
    }
)


def is_read_only_tool(tool: StructuredTool) -> bool:
    """Use explicit metadata; unknown tools default to non-retryable writes."""
    return bool((tool.metadata or {}).get("read_only", False))


class WorkoutExerciseInput(BaseModel):
    """Concrete JSON schema for one exercise accepted by GigaChat tool calling."""

    name: str = Field(description="Exercise name")
    sets: int | None = Field(default=None, ge=1, description="Number of sets")
    reps: str | None = Field(
        default=None,
        description="Repetitions, for example '10' or '8-12'",
    )
    weight_kg: float | None = Field(default=None, ge=0, description="Used weight in kg")
    notes: str | None = Field(default=None, description="Optional exercise notes")


def build_tools(user_id: str, domains: Iterable[ToolDomain] | None = None) -> list[StructuredTool]:
    """Возвращает инструменты, привязанные к одному пользователю.

    domains позволяет специалистам получать только свои инструменты:
    Nutrition Agent не видит запись тренировок, Recovery Agent не пишет еду.
    """
    enabled = set(domains or ("profile", "nutrition"))
    tools: list[StructuredTool] = []

    def get_my_profile() -> dict:
        return profile_tools.get_profile(user_id)

    def search_food(query: str) -> dict:
        return nutrition_tools.search_food(query)

    def get_daily_intake(day: str | None = None) -> dict:
        return nutrition_tools.get_daily_intake(user_id, day)

    def log_meal(
        name: str,
        calories: float,
        protein_g: float,
        carbs_g: float,
        fat_g: float,
        meal_type: str | None = None,
        day: str | None = None,
    ) -> dict:
        return nutrition_tools.log_meal(
            user_id, name, calories, protein_g, carbs_g, fat_g, meal_type, day
        )

    def get_workout_history(days: int = 14) -> dict:
        return workout_tools.get_workout_history(user_id, days)

    def log_workout(
        workout_type: str,
        duration_min: float | None = None,
        exercises: list[WorkoutExerciseInput] | None = None,
        calories_burned: float | None = None,
        notes: str | None = None,
        day: str | None = None,
    ) -> dict:
        exercise_rows = [exercise.model_dump(exclude_none=True) for exercise in exercises or []]
        return workout_tools.log_workout(
            user_id,
            workout_type,
            duration_min,
            exercise_rows,
            calories_burned,
            notes,
            day,
        )

    def get_recovery_logs(days: int = 14) -> dict:
        return recovery_tools.get_recovery_logs(user_id, days)

    def get_weight_trend(days: int = 30) -> dict:
        return recovery_tools.get_weight_trend(user_id, days)

    def get_cycle_logs(days: int = 45) -> dict:
        return calendar_tools.get_cycle_logs(user_id, days)

    if "profile" in enabled:
        tools.append(StructuredTool.from_function(
            func=get_my_profile,
            name="get_my_profile",
            metadata={"read_only": True},
            description=(
                "Профиль пользователя: возраст, пол, рост, вес, цель, "
                "целевые калории и БЖУ, аллергии, предпочтения. "
                "Вызывай перед персональным советом."
            ),
        ))

    if "nutrition" in enabled:
        tools.extend([
            StructuredTool.from_function(
                func=search_food,
                name="search_food",
                metadata={"read_only": True},
                description=(
                    "Ищет продукт в справочнике и возвращает его КБЖУ на 100 г. "
                    "Аргумент query — название продукта, например 'куриная грудка'."
                ),
            ),
            StructuredTool.from_function(
                func=get_daily_intake,
                name="get_daily_intake",
                metadata={"read_only": True},
                description=(
                    "Показывает, что пользователь УЖЕ съел за день: суммы КБЖУ "
                    "и список приёмов пищи. day — ГГГГ-ММ-ДД, по умолчанию сегодня."
                ),
            ),
            StructuredTool.from_function(
                func=log_meal,
                name="log_meal",
                metadata={"read_only": False},
                description=(
                    "ЗАПИСЫВАЕТ приём пищи в дневник. Вызывай только когда "
                    "пользователь явно просит записать съеденное. "
                    "meal_type: breakfast, lunch, dinner или snack."
                ),
            ),
        ])

    if "workout" in enabled:
        tools.extend([
            StructuredTool.from_function(
                func=get_workout_history,
                name="get_workout_history",
                metadata={"read_only": True},
                description="История тренировок пользователя за последние days дней для прогрессии и нагрузки.",
            ),
            StructuredTool.from_function(
                func=log_workout,
                name="log_workout",
                metadata={"read_only": False},
                description=(
                    "ЗАПИСЫВАЕТ тренировку. Вызывай только после явной просьбы пользователя. "
                    "workout_type: upper_body, lower_body, full_body, functional, crossfit, cardio или rest."
                ),
            ),
        ])

    if "recovery" in enabled:
        tools.extend([
            StructuredTool.from_function(
                func=get_recovery_logs,
                name="get_recovery_logs",
                metadata={"read_only": True},
                description="Сон, энергия, настроение и симптомы за последние days дней.",
            ),
            StructuredTool.from_function(
                func=get_weight_trend,
                name="get_weight_trend",
                metadata={"read_only": True},
                description="Записи веса за последние days дней и изменение веса за период.",
            ),
        ])

    if "calendar" in enabled:
        tools.append(StructuredTool.from_function(
            func=get_cycle_logs,
            name="get_cycle_logs",
            metadata={"read_only": True},
            description="Opt-in записи цикла за последние days дней для recovery/cycle-aware советов.",
        ))

    return tools
