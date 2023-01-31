import requests


class DatastoreWrapper():
    def __init__(self, url):
        self.url = url

    def create_session(self, data: dict) -> (str, int):
        resp = requests.post(f'{self.url}/session/', data=data)
        if resp.status_code == 201:
            json = resp.json()
            return json['session_name'], json['id']
        else:
            raise Exception(resp.status_code)

    def buy_crypto(self, session: str, step: int, amount: float, value: float):
        pass

    def sell_crypto(self, session: str, step: int, amount: float, value: float):
        pass

    def get_session(self, session_name: str):
        pass

    def get_all_sessions(self, ) -> list:
        pass

    def get_session_by_id(self, session_id: int) -> dict:
        pass

    def get_session_transactions(self, session_name: str = None, session_id: int = None) -> list:
        if session_name is None and session_id is None:
            raise NameError


if __name__ == '__maine__':
    client = DatastoreWrapper('localhost:2522')