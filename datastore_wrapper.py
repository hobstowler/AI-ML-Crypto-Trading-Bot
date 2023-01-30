import requests

def create_session() -> (str, int):
    pass

def buy_crypto(session: str, step: int, amount: float, value: float):
    pass

def sell_crypto(session: str, step: int, amount: float, value: float):
    pass

def get_session(session_name: str):
    pass

def get_all_sessions() -> list:
    pass

def get_session_by_id(session_id: int) -> dict:
    pass

def get_session_transactions(session_name: str = None, session_id: int = None) -> list:
    if session_name is None and session_id is None:
        raise NameError

