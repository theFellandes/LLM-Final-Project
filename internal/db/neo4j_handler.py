from neo4j import GraphDatabase


class Neo4jHandler:
    """
    Handles Neo4j interactions for nodes and relationships.
    """

    def __init__(self, connection_tuple):
        uri, user, password = connection_tuple
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        """Closes the Neo4j driver."""
        self.driver.close()

    def create_node(self, label, properties):
        """
        Creates or updates a node with the given label and properties.
        """
        query = f"""
        MERGE (n:{label} {{id: $id}})
        SET n += $properties
        """
        self._execute_query(query, {"id": properties["id"], "properties": properties})

    def create_relationship(self, from_node, to_node, relationship, properties=None):
        """
        Creates or updates a relationship between two nodes.
        """
        query = f"""
        MATCH (a {{id: $from_id}}), (b {{id: $to_id}})
        WHERE EXISTS((a)) AND EXISTS((b))
        MERGE (a)-[r:{relationship}]->(b)
        SET r += $properties
        """
        self._execute_query(query, {
            "from_id": from_node["id"],
            "to_id": to_node["id"],
            "properties": properties or {}
        })

    def _execute_query(self, query, parameters):
        """
        Executes a query with the provided parameters.
        """
        with self.driver.session() as session:
            session.run(query, parameters)

