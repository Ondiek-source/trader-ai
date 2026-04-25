from ..http.navigator import Browser


class Settings(Browser):

    def __init__(self, api):
        super().__init__()
        self.api = api

    def get_settings(self):
        """Fetch the cabinet digest (user settings/status)."""
        self.headers.update(
            {
                "content-type": "application/json",
                "referer": f"{self.api.https_url}/{self.api.lang}/trade",
                "cookie": self.api.session_data.get("cookies", ""),
                "user-agent": self.api.session_data.get("user_agent", ""),
            }
        )
        # We call send_request, which updates self.response internally
        self.send_request("GET", f"{self.api.https_url}/api/v1/cabinets/digest")

        # return the parsed JSON from the response
        return self.get_json()

    def set_time_offset(self, time_offset: int):
        payload = {"time_offset": time_offset}
        self.headers.update(
            {
                "referer": f"{self.api.https_url}/{self.api.lang}/trade",
                "cookie": self.api.session_data.get("cookies", ""),
                "user-agent": self.api.session_data.get("user_agent", ""),
            }
        )

        self.send_request(
            method="POST",
            url=f"{self.api.https_url}/api/v1/user/profile/time_offset",
            json=payload,
        )

        # return the result of the POST request
        return self.get_json()
