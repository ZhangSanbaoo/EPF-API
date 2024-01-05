import google.auth
from google.cloud import firestore
from google.oauth2 import service_account

class FirestoreClient:
    """Wrapper around a database"""

    client: firestore.Client

    def __init__(self) -> None:
        """Init the client."""
        key_path="services/epf-flower-data-science/src/config/epf-api-4f7c8-firebase-adminsdk-hzb9e-bce0b4d8cb.json"
        credentials = service_account.Credentials.from_service_account_file(key_path)

        self.client = firestore.Client(credentials=credentials)

    def get(self, collection_name: str, document_id: str) -> dict:
        """Find one document by ID.
        Args:
            collection_name: The collection name
            document_id: The document id
        Return:
            Document value.
        """
        doc = self.client.collection(
            collection_name).document(document_id).get()
        if doc.exists:
            return doc.to_dict()
        raise FileExistsError(
            f"No document found at {collection_name} with the id {document_id}"
        )

    def update(self, collection_name: str, document_id: str, data: dict) -> None:
        self.client.collection(collection_name).document(document_id).update(data)