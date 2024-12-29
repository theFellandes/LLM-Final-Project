import csv
from neo4j import GraphDatabase
from SPARQLWrapper import SPARQLWrapper, JSON

uri = "bolt://localhost:7687"
user = "neo4j"
password = "your_password"
driver = GraphDatabase.driver(uri, auth=(user, password))


def store_award_in_neo4j(book_id, author_name, award_name):
    with driver.session() as session:
        session.run(
            """
            MERGE (b:Book {id: $book_id})
            MERGE (a:Author {name: $author_name})
            MERGE (aw:HAS_AWARDS {name: $award_name})
            MERGE (b)-[:HAS_AWARDS]->(aw)
            MERGE (a)-[:HAS_AWARDS]->(aw)
            """,
            book_id=book_id,
            author_name=author_name,
            award_name=award_name
        )


# 1. DBpedia SPARQL setup
sparql = SPARQLWrapper("https://dbpedia.org/sparql")
sparql.setReturnFormat(JSON)

# 2. Read your CSV (book1-100k.csv) with columns:
# Id, Name, ..., ISBN, ..., Authors
# Adjust indexes as needed
csv_file = "../../../.cache/kagglehub/datasets/bahramjannesarr/goodreads-book-datasets-10m/versions/18/book1-100k.csv"
with open(csv_file, "r", newline="", encoding="utf-8") as infile:
    reader = csv.reader(infile)
    header = next(reader)  # skip header

    for row in reader:
        book_id = row[0].strip()  # "Id"
        book_name = row[1].strip()  # "Name"
        isbn_value = row[16].strip()  # "ISBN" (17th column)
        authors_str = row[12].strip()  # "Authors" (13th column)

        if not isbn_value:
            # skip if no ISBN
            continue

        # Possibly you have multiple authors in "Authors" column (e.g. "J.K. Rowling, Mary GrandPré")
        # For simplicity, let's just take the first author
        author_name = authors_str.split(",")[0].strip() if authors_str else "Unknown Author"

        # 3. Query DBpedia for awards
        query = f"""
        PREFIX dbo: <http://dbpedia.org/ontology/>
        PREFIX dbp: <http://dbpedia.org/property/>
        SELECT ?award ?award2
        WHERE {{
          ?book rdf:type dbo:Book .
          OPTIONAL {{ ?book dbo:isbn ?isbn FILTER (?isbn = "{isbn_value}") }}
          OPTIONAL {{ ?book dbo:award ?award }}
          OPTIONAL {{ ?book dbp:awards ?award2 }}
        }}
        """
        sparql.setQuery(query)

        try:
            results = sparql.query().convert()
        except Exception as e:
            print(f"Failed querying DBpedia for ISBN={isbn_value}: {e}")
            continue

        # 4. Parse awards out of the SPARQL results
        #    awards might be URIs or strings, depending on DBpedia data
        for binding in results["results"]["bindings"]:
            # dbo:award might be in binding["award"]["value"]
            # dbp:awards might be in binding["award2"]["value"]
            if "award" in binding and binding["award"]["value"]:
                award_value = binding["award"]["value"]
                # If it's a DBpedia URI, you might want to extract a label
                # or you can store the URI directly
                store_award_in_neo4j(book_id, author_name, award_value)

            if "award2" in binding and binding["award2"]["value"]:
                award_value_2 = binding["award2"]["value"]
                store_award_in_neo4j(book_id, author_name, award_value_2)

print("Done storing awards in Neo4j.")
driver.close()
