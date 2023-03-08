from google.cloud import secretmanager

class Project:
    def __init__(self, project_id=None) -> None:
        self.secret_client = secretmanager.SecretManagerServiceClient()

        self.project_id = '638911377324' if project_id is None else project_id

    def get_secret(self, secret_name, version='latest') -> str:
        response = self.secret_client.access_secret_version(request={
            'name': f'projects/{self.project_id}/secrets/{secret_name}/versions/{version}'
        })
        return response.payload.data.decode('UTF-8')

class BinanceKeys:
    
    def __init__(self) -> None:
        ProjectSecrets = Project()
        self._api_key = ProjectSecrets.get_secret('binance-api-key-ken')
        self._api_secret = ProjectSecrets.get_secret('binance-api-secret-ken')
        
    def get_api_key(self):
        return self._api_key
    
    def get_api_secret(self):
        return self._api_secret
