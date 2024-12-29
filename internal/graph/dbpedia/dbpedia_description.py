import csv
from neo4j import GraphDatabase
from SPARQLWrapper import SPARQLWrapper, JSON

# -----------------------
# Neo4j Connection Setup
# -----------------------
URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "your_password"

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

# -----------------------
# DBpedia SPARQL Setup
# -----------------------
sparql = SPARQLWrapper("https://dbpedia.org/sparql")
sparql.setReturnFormat(JSON)


def build_sparql_query(isbn_value: str, book_title: str) -> str:
    """
    Build a SPARQL query that tries to match either:
    - `dbo:isbn isbn_value`
    OR
    - `rdfs:label book_title@en`
    Then tries dbo:abstract or rdfs:comment in English.
    """
    return f"""
    PREFIX dbo: <http://dbpedia.org/ontology/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT ?abstract ?comment
    WHERE {{
      ?book rdf:type dbo:Book .
      {{
        ?book dbo:isbn "{isbn_value}"
      }}
      UNION
      {{
        ?book rdfs:label "{book_title}"@en
      }}

      OPTIONAL {{
        ?book dbo:abstract ?abstract .
        FILTER (lang(?abstract) = "en")
      }}
      OPTIONAL {{
        ?book rdfs:comment ?comment .
        FILTER (lang(?comment) = "en")
      }}
    }}
    LIMIT 1
    """


def fetch_description(isbn_value: str, book_title: str) -> str:
    """
    Run SPARQL to get `dbo:abstract` if available, or fallback to `rdfs:comment`.
    Matches either on ISBN or on the given title in English.
    """
    query = build_sparql_query(isbn_value, book_title)
    sparql.setQuery(query)

    try:
        results = sparql.query().convert()
    except Exception as e:
        print(f"[ERROR] DBpedia query failed: ISBN={isbn_value}, title={book_title}. {e}")
        return None

    for binding in results["results"]["bindings"]:
        # Prefer abstract first
        if "abstract" in binding and binding["abstract"]["value"].strip():
            return binding["abstract"]["value"].strip()
        # Fallback to comment
        if "comment" in binding and binding["comment"]["value"].strip():
            return binding["comment"]["value"].strip()
    return None


def main():
    # Example CSV path
    csv_file = "../../../.cache/kagglehub/datasets/bahramjannesarr/goodreads-book-datasets-10m/versions/18/book1-100k.csv"

    # We assume:
    #  - Column[0] => Book ID
    #  - Column[1] => Book Title
    #  - Column[16] => ISBN
    with driver.session() as session, open(csv_file, "r", encoding="utf-8") as infile:
        reader = csv.reader(infile)
        header = next(reader, None)  # skip header row if any

        for row_num, row in enumerate(reader, start=2):
            if len(row) < 17:
                continue

            book_id = row[0].strip()
            book_title = row[1].strip()
            isbn_value = row[16].strip()

            if not book_title and not isbn_value:
                # If there's literally no ISBN or title, skip
                continue

            # Fetch from DBpedia
            description = fetch_description(isbn_value, book_title)
            if description:
                snippet = description[:80].replace("\n", " ")
                print(f"[ROW {row_num}] Book ID={book_id}, found desc snippet: \"{snippet}\"...")

                session.run("""
                    MERGE (b:Book {id: $book_id})
                    SET b.description = $description
                    """,
                            book_id=book_id,
                            description=description
                            )
            else:
                print(f"[ROW {row_num}] Book ID={book_id}, no description found.")


if __name__ == "__main__":
    try:
        main()
    finally:
        driver.close()
        print("Neo4j driver closed.")
