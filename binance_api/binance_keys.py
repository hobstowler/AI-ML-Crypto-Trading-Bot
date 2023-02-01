# these keys are hard coded for Tyler's personal account for now, we should swap these out
# when we have a project account set up
API_KEY = "DJvRLFOnlYgSKMFHzigoF2QGBL1AZXz0BsGm8ML168mC8zSoSOjOkq3GT5NhSRll"
API_SECRET = "2SA3MkbKncuKZ7aLbNfw7BZpUOelPAagNSAfKFgJF8uIkra1EvSB3wzBmBx46KCt" 


class BinanceKeys:
    
    def __init__(self) -> None:
        self._api_key = API_KEY
        self._api_secret = API_SECRET
        
    def get_api_key(self):
        return self._api_key
    
    def get_api_secret(self):
        return self._api_secret
