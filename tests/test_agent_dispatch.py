"""Tests basiques pour l'agent dispatch intelligent (Phase 1)."""
import pytest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from services.agent_dispatch.orchestrator import AgentOrchestrator, get_agent_for_company
from services.agent_dispatch.tools import AgentTools
from models import Company


@pytest.fixture
def sample_company(db_session):
    """Créer une entreprise de test."""
    company = Company(
        name="Test Company Agent",
        email="test@agent.com",
        phone="+41123456789",
    )
    db_session.add(company)
    db_session.commit()
    return company


def test_agent_orchestrator_initialization(sample_company):
    """Test que l'orchestrateur peut être initialisé."""
    agent = AgentOrchestrator(sample_company.id)
    assert agent.company_id == sample_company.id
    assert agent.state.running is False
    assert agent.tools is not None


def test_agent_tools_get_state(sample_company, db_session):
    """Test que get_state retourne un format correct."""
    tools = AgentTools(sample_company.id)
    
    window_start = datetime.now(ZoneInfo("Europe/Zurich"))
    window_end = window_start + timedelta(hours=2)
    
    state = tools.get_state(window_start, window_end)
    
    assert "drivers" in state
    assert "jobs" in state
    assert "constraints" in state
    assert "slas" in state
    assert isinstance(state["drivers"], list)
    assert isinstance(state["jobs"], list)


def test_agent_tools_osrm_health(sample_company):
    """Test que osrm_health retourne un format correct."""
    tools = AgentTools(sample_company.id)
    
    health = tools.osrm_health()
    
    assert "state" in health
    assert "latency_ms" in health
    assert "fail_ratio" in health
    assert health["state"] in ["CLOSED", "OPEN", "HALF_OPEN"]


def test_agent_tools_log_action(sample_company, db_session):
    """Test que log_action crée une entrée dans AutonomousAction."""
    from models.autonomous_action import AutonomousAction
    
    tools = AgentTools(sample_company.id)
    
    result = tools.log_action(
        kind="test",
        payload={"test": "data"},
        reasoning_brief="Test action",
    )
    
    assert "event_id" in result
    
    # Vérifier que l'action a été créée
    action = (
        db_session.query(AutonomousAction)
        .filter(AutonomousAction.company_id == sample_company.id)
        .filter(AutonomousAction.action_type == "test")
        .first()
    )
    
    assert action is not None
    assert action.trigger_source == "agent_dispatch"
    assert action.success is True


def test_get_agent_for_company(sample_company):
    """Test que get_agent_for_company retourne un agent."""
    agent = get_agent_for_company(sample_company.id)
    assert agent is not None
    assert agent.company_id == sample_company.id
    
    # Test singleton : même instance
    agent2 = get_agent_for_company(sample_company.id)
    assert agent is agent2


def test_agent_get_status(sample_company):
    """Test que get_status retourne un format correct."""
    agent = AgentOrchestrator(sample_company.id)
    
    status = agent.get_status()
    
    assert "running" in status
    assert "last_tick" in status
    assert "actions_today" in status
    assert "actions_last_hour" in status
    assert "osrm_health" in status
    assert status["running"] is False

