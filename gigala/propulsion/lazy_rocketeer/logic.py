import langchain_openai
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from config import OPENAI_API_KEY


class DialogModel:
    openai_client= None
    _model= None

    def __init__(self, model, api_key_):
        self._model = model
        self._key = api_key_
        self.openai_client = langchain_openai.ChatOpenAI(model = self._model,
                                                         openai_api_key = self._key,
                                                         temperature = 0)


class ChamberInput(BaseModel):
    r: float = Field(..., description="Mixture ratio (O/F)")
    F: float  = Field(..., description="Thrust,[N]")
    p1: float = Field(..., description="Chamber pressure, [MPa]")
    CF: float = Field(..., description="Thrust coefficient")
    c: float = Field(..., description="Estimated nozzle exit exhaust velocity, [m/sec]")
    m_p: float = Field(..., description="Usable propellant mass, [kg]")

@tool(args_schema=ChamberInput)
def get_thrust_chamber_params(r, F, p1, CF, c, m_p):
    """ Thrust chamber dimensions and burn duration calculations.
        r = 2.3
        F = 50000
        p1 = 4826000
        CF = 1.9
        m_p = 7482
        """
    m_hat = F/c
    m_hat_f =m_hat/(r+1)
    m_hat_o = (m_hat*r) / (r + 1)
    t_b = m_p /(m_hat_f+m_hat_o)
    A_t = F/(p1*CF)
    return {"nozzle_throat_area": A_t,
            "burn_duration":t_b,
            }

MODEL = DialogModel("gpt-4o-mini", OPENAI_API_KEY)
