import unittest

from data.datastore_wrapper import DatastoreWrapper


class MyTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        url = 'https://datastore-micro-dot-ai-ml-bitcoin-bot.uw.r.appspot.com/'
        cls.wrapper = DatastoreWrapper(url)

    def test_create_and_delete_session(self):
        num_sessions = len(self.wrapper.get_all_sessions())
        session_id = self.wrapper.create_session(**{
            "session_name": "test session",
            "type": "test type",
            "starting_balance": 1000.0,
            "starting_coins": 10.0,
            "crypto_type": "test crypto"
        })
        num_sessions2 = len(self.wrapper.get_all_sessions())
        self.assertEqual(num_sessions + 1, num_sessions2)
        self.assertTrue(self.wrapper.delete_session(session_id))
        num_sessions = len(self.wrapper.get_all_sessions())
        self.assertEqual(num_sessions2 - 1, num_sessions)

    def test_edit_session(self):
        name = "test session"
        end_bal = 89.0
        session_id = self.wrapper.create_session(**{
            "session_name": name,
            "type": "test type",
            "starting_balance": 1000.0,
            "starting_coins": 10.0,
            "crypto_type": "test crypto"
        })
        session = self.wrapper.get_session_by_id(session_id)
        self.assertEqual(session['session_name'], name)
        self.wrapper.edit_session(session_id, **{
            "ending_balance": end_bal
        })
        session = self.wrapper.get_session_by_id(session_id)
        self.assertEqual(session['session_name'], name)  # shouldn't change
        self.assertEqual(session['ending_balance'], end_bal)
        self.assertTrue(self.wrapper.delete_session(session_id))


if __name__ == '__main__':
    unittest.main()
