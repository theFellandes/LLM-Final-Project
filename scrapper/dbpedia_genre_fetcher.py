from SPARQLWrapper import SPARQLWrapper, JSON


class DBpediaHandler:
    def __init__(self, sparql_endpoint="http://dbpedia.org/sparql"):
        self.sparql = SPARQLWrapper(sparql_endpoint)

    @staticmethod
    def sanitize_title(title):
        """
        Sanitize the title to escape special characters for SPARQL queries.
        """
        return title.replace('"', '\\"').replace("'", "\\'")

    def fetch_book_metadata(self, book_title):
        """
        Fetch metadata for a book from DBpedia.
        """
        sanitized_title = self.sanitize_title(book_title["name"])
        query = f"""
        PREFIX dbo: <http://dbpedia.org/ontology/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX dct: <http://purl.org/dc/terms/>

        SELECT ?s (SAMPLE(?desc) AS ?label) 
               (GROUP_CONCAT(?authorlabel, ', ') AS ?author) 
               (GROUP_CONCAT(?genrelabel, ', ') AS ?genre)
        WHERE {{
          ?s rdf:type dbo:Book .
          OPTIONAL {{ ?s dbo:author ?author . }}
          OPTIONAL {{ ?s rdfs:label ?desc FILTER (lang(?desc) = "en"). }}
          OPTIONAL {{ ?author rdfs:label ?authorlabel FILTER (lang(?authorlabel) = "en"). }}
          OPTIONAL {{
            ?s dct:subject ?subject .
            ?subject rdfs:label ?genrelabel FILTER (lang(?genrelabel) = "en").
          }}
          OPTIONAL {{
            ?s dbo:literaryGenre ?genre .
            ?genre rdfs:label ?genrelabel FILTER (lang(?genrelabel) = "en").
          }}
          FILTER (
            CONTAINS(lcase(str(?desc)), lcase("{sanitized_title}")) || 
            REGEX(lcase(str(?desc)), "{sanitized_title}", "i")
          )
        }} GROUP BY ?s
        """
        self.sparql.setQuery(query)
        self.sparql.setReturnFormat(JSON)

        try:
            results = self.sparql.query().convert()
            bindings = results.get("results", {}).get("bindings", [])
            if bindings:
                return {
                    "description": bindings[0].get("label", {}).get("value", None),
                    "author": bindings[0].get("author", {}).get("value", None),
                    "genre": bindings[0].get("genre", {}).get("value", None),
                }
        except Exception as e:
            print(f"Error querying DBpedia for '{book_title}': {e}")
        return None
