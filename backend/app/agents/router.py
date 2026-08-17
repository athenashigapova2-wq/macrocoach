"""Router Agent: chooses the specialist agent for the next turn."""

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.prompts import ROUTER_SYSTEM
from app.agents.state import AgentName, AgentState
from app.llm import get_routed_llm
from app.services import agent_traces

_ALLOWED: set[AgentName] = {"nutrition", "workout", "recovery", "general"}

_KEYWORDS: dict[AgentName, tuple[str, ...]] = {
    "nutrition": (
        "калор", "кбжу", "бел", "жир", "углев", "еда", "съел",
        "meal", "food", "protein", "calorie", " ate ",
        "repas", "aliment", "protéin", "calorie",
        "comida", "alimento", "proteína", "caloría",
        "食物", "吃", "卡路里", "蛋白质", "碳水", "脂肪",
    ),
    "workout": (
        "трен", "зал", "упраж", "подход", "присед", "выпад",
        "workout", "gym", "exercise", "sets", "reps",
        "entraînement", "exercice", "salle", "série",
        "entrenamiento", "ejercicio", "gimnasio", "serie",
        "训练", "健身房", "运动", "组数", "重复",
    ),
    "recovery": (
        "сон", "спал", "спала", "устал", "цикл", "вес", "болит", "болят",
        "sleep", "fatigue", "cycle", "sore", "weight",
        "sommeil", "fatigu", "poids", "douleur", "récupération",
        "sueño", "fatiga", "ciclo", "peso", "dolor", "recuperación",
        "睡眠", "疲劳", "月经", "周期", "体重", "疼", "恢复",
    ),
}


def _last_user_text(state: AgentState) -> str:
    for message in reversed(state["messages"]):
        if getattr(message, "type", None) == "human":
            return str(message.content)
    return ""


def route_with_keywords(text: str) -> AgentName:
    """Deterministic fallback used when the LLM router is unavailable."""
    lowered = text.lower()
    scores = {
        route: sum(1 for keyword in keywords if keyword in lowered)
        for route, keywords in _KEYWORDS.items()
    }
    best_route, best_score = max(scores.items(), key=lambda item: item[1])
    return best_route if best_score > 0 else "general"


def router_node(state: AgentState) -> dict[str, AgentName]:
    """LangGraph node that writes `route` into the state."""
    text = _last_user_text(state)
    try:
        llm, selection = get_routed_llm(
            node_name="router",
            purpose="route_classification",
            default_tier="small",
            temperature=0.0,
        )
        response = agent_traces.invoke_llm(
            llm,
            [SystemMessage(content=ROUTER_SYSTEM), HumanMessage(content=text)],
            run_id=state.get("run_id"),
            node_name="router",
            purpose="route_classification",
            model_tier=selection.model_tier,
            model_selection=selection,
        )
        route = str(response.content).strip().lower().split()[0]
        if route in _ALLOWED:
            return {"route": route}  # type: ignore[return-value]
    except Exception:
        pass
    return {"route": route_with_keywords(text)}
