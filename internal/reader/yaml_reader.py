from pydantic import BaseModel, Field, field_validator
import yaml
import os


class YAMLReader(BaseModel):
    yaml_file_path: str = Field(..., description="Path to the YAML file")

    yaml_data: dict = Field(default_factory=dict, description="Parsed YAML data")

    @field_validator("yaml_file_path")
    def file_exists(cls, value):
        """
        Validates if the YAML file exists.
        """
        if not os.path.exists(value):
            raise FileNotFoundError(f"YAML file not found at: {value}")
        return value

    def load_yaml(self):
        """
        Loads and parses the YAML file.
        """
        try:
            with open(self.yaml_file_path, 'r') as file:
                self.yaml_data = yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")

    def get(self, *keys):
        """
        Retrieves a value from the YAML data using a sequence of keys.
        """
        data = self.yaml_data
        for key in keys:
            data = data.get(key)
            if data is None:
                return None
        return data

    def has_key(self, *keys):
        """
        Checks if a sequence of keys exists in the YAML data.
        """
        data = self.yaml_data
        for key in keys:
            if key not in data:
                return False
            data = data[key]
        return True
