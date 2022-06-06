import json


class Config(object):
    """Base class for experimental setting/configuration."""

    def __init__(self, settings):
        self.settings = settings

    def load_config(self, import_json):
        """Load settings dict from import_json (path/filename.json) JSON-file."""

        with open(import_json, 'r') as fp:
            settings = json.load(fp)

        for key, value in settings.items():
            self.settings[key] = value

    def save_config(self, export_json):
        """Save settings dict to export_json (path/filename.json) JSON-file."""

        with open(export_json, 'w') as fp:
            json.dump(self.settings, fp)
