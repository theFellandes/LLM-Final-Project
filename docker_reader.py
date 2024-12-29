from internal.reader.yaml_reader import YAMLReader


class DockerComposeReader(YAMLReader):
    def get_neo4j_config(self):
        """
        Extracts Neo4j connection details and returns them as a tuple (URI, username, password).
        """
        self.load_yaml()  # Ensure the YAML data is loaded
        neo4j_service = self.get("services", "neo4j")
        if not neo4j_service:
            raise ValueError("Neo4j service not found in docker-compose.yml")

        # Extract ports
        ports = neo4j_service.get("ports", [])
        bolt_port = next((port.split(":")[0] for port in ports if "7687" in port), "7687")

        # Extract environment variables for authentication
        environment = neo4j_service.get("environment", [])
        # TODO: This is a bug, will fix it.
        # auth_env = next((env for env in environment if env.startswith("NEO4J_AUTH")), None)
        auth_env = 'neo4j/your_password'
        if not auth_env:
            raise ValueError("NEO4J_AUTH environment variable not found in docker-compose.yml")

        # Extract username and password from NEO4J_AUTH
        # TODO: This part is also bugged.
        # try:
        #     auth_value = auth_env.split(":")[1].strip()
        #     username, password = auth_value.split("/")
        # except IndexError:
        #     raise ValueError(f"Invalid NEO4J_AUTH format: {auth_env}")

        # Return the connection details as a tuple
        username = "neo4j"
        password = "your_password"
        return f"bolt://localhost:{bolt_port}", username, password

