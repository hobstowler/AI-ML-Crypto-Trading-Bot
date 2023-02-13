import requests


class DatastoreWrapper():
    def __init__(self, url=None):
        if url is None:
            url = 'https://datastore-micro-dot-ai-ml-bitcoin-bot.uw.r.appspot.com/'
        self.url = url

    def create_session(self, **kwargs) -> int:
        """
        Creates a new session entity with the provided keyword arguments.
        Returns the resulting session_id from datastore to use in later requests.
        Required keyword arguments and types:
            session_name : str
            type : str
            starting_balance : float
            starting_coins : float
            crypto_type : str
        Refer to README for full list of keyword arguments.
        :param kwargs: Session details to be sent in the request.
        :return: The session_id upon success.
        """
        # validate inputs
        if 'session_name' not in kwargs:
            raise Exception('Missing required "session_name" in kwargs.')
        if 'type' not in kwargs:
            raise Exception('Missing required "type" in kwargs.')
        if 'starting_balance' not in kwargs:
            raise Exception('Missing required "starting_balance" in kwargs.')
        if 'starting_coins' not in kwargs:
            raise Exception('Missing required "starting_coins" in kwargs.')
        if 'crypto_type' not in kwargs:
            raise Exception('Missing required "crypto_type" in kwargs.')

        resp = requests.post(f'{self.url}/session/', json=kwargs)
        if resp.status_code == 201:
            json = resp.json()
            return int(json['id'])
        else:
            print(f'response from server: {resp.status_code}')
            raise Exception(f'response from server: {resp.status_code}')

    def get_all_sessions(self) -> list:
        resp = requests.get(f'{self.url}/session')
        if resp.status_code == 200:
            json = resp.json()['sessions']
            return json
        else:
            raise Exception(f'response from server: {resp.status_code}')

    def get_session_by_id(self, session_id: int) -> dict:
        resp = requests.get(f'{self.url}/session/{session_id}')
        if resp.status_code == 200:
            json = resp.json()
            return json
        else:
            raise Exception(f'response from server: {resp.status_code}')

    def edit_session(self, session_id: int, **kwargs):
        resp = requests.patch(f'{self.url}/session/{session_id}', json=kwargs)
        if resp.status_code == 200:
            json = resp.json()
            return json
        else:
            raise Exception(f'response from server: {resp.status_code}')

    def delete_session(self, session_id: int):
        resp = requests.delete(f'{self.url}/session/{session_id}')
        if resp.status_code == 204:
            return True
        else:
            return False

    def get_session_transactions(self, session_id: int) -> list:
        resp = requests.get(f'{self.url}/session/{session_id}/transaction')
        if resp.status_code == 200:
            json = resp.json()
            return json
        else:
            raise Exception(f'response from server: {resp.status_code}')

    def buy_crypto(self, session: str, step: int, amount: float, value: float):
        pass

    def sell_crypto(self, session: str, step: int, amount: float, value: float):
        pass


if __name__ == '__main__':
    client = DatastoreWrapper('https://datastore-micro-dot-ai-ml-bitcoin-bot.uw.r.appspot.com/')
    print(client.get_all_sessions())
    print(client.get_session_by_id(5634161670881280))

    sesh_id = client.create_session(**{
        "session_name": "test session",
        "type": "test type",
        "starting_balance": 1000.0,
        "starting_coins": 10.0,
        "crypto_type": "test crypto"
    })
    print(sesh_id)
    print(client.delete_session(sesh_id))
