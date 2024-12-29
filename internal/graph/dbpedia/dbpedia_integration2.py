import csv
from neo4j import GraphDatabase
from SPARQLWrapper import SPARQLWrapper, JSON

# Step 1: Connect to Neo4j
uri = "bolt://localhost:7687"
user = "neo4j"
password = "your_password"
driver = GraphDatabase.driver(uri, auth=(user, password))

# Step 2: Define a helper function to store Book & Genre in Neo4j
def store_in_neo4j(book_id, isbn, genre):
    with driver.session() as session:
        session.run(
            """
            MERGE (b:Book {id: $book_id})
              ON CREATE SET b.isbn = $isbn
              ON MATCH SET b.isbn = $isbn
            MERGE (g:Genre {name: $genre})
            MERGE (b)-[:BELONGS_TO]->(g)
            """,
            book_id=book_id,
            isbn=isbn,
            genre=genre
        )

# Step 3: Set up your DBpedia SPARQL endpoint
sparql = SPARQLWrapper("https://dbpedia.org/sparql")
sparql.setReturnFormat(JSON)

# Step 4: Read your CSV and process each row
csv_file = "../../../.cache/kagglehub/datasets/bahramjannesarr/goodreads-book-datasets-10m/versions/18/book1-100k.csv"

with open(csv_file, "r", newline="", encoding="utf-8") as infile:
    reader = csv.reader(infile)
    header = next(reader)  # skip header row

    for row in reader:
        # The CSV columns (0-based index):
        # 0: Id
        # ...
        # 16: ISBN
        book_id = row[0].strip()
        isbn_value = row[16].strip() if len(row) > 16 else ""

        if not isbn_value:
            # Skip if ISBN is empty
            continue

        # Build the DBpedia query
        query = f"""
        PREFIX dbo: <http://dbpedia.org/ontology/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX dct: <http://purl.org/dc/terms/>

        SELECT ?genre
        WHERE {{
          ?book rdf:type dbo:Book .
          OPTIONAL {{ ?book dbo:isbn ?isbn FILTER (?isbn = "{isbn_value}") }}
          OPTIONAL {{ ?book dct:subject/rdfs:label ?genre FILTER (lang(?genre) = "en") }}
        }}
        LIMIT 1
        """

        sparql.setQuery(query)
        try:
            results = sparql.query().convert()
        except Exception as e:
            print(f"Failed to fetch DBpedia data for ISBN={isbn_value}: {e}")
            continue

        # Extract any genre from the query results
        genre_found = None
        for binding in results["results"]["bindings"]:
            if "genre" in binding:
                genre_found = binding["genre"]["value"]
                break  # we only take the first one if multiple appear

        # If a genre is found, store it in Neo4j
        if genre_found:
            store_in_neo4j(book_id, isbn_value, genre_found)
        else:
            # Optionally, store the fact that no genre was found
            # or simply do nothing
            pass

print("Done storing Book-Genre data in Neo4j.")
