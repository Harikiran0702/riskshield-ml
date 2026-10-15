import joblib
import pytest
from fastapi.testclient import TestClient

from riskshield.api.main import create_app
from riskshield.api.schemas import Transaction
from riskshield.api.security import ServiceSettings
from riskshield.data.generator import generate
from riskshield.models.training import train


@pytest.fixture(scope="session")
def frame():
    return generate(3500)


@pytest.fixture(scope="session")
def trained(tmp_path_factory, frame):
    root = tmp_path_factory.mktemp("training")
    data = root / "transactions.csv"
    frame.to_csv(data, index=False)
    train(data, root / "artifacts", root / "reports")
    return root


@pytest.fixture(scope="session")
def bundle(trained):
    return joblib.load(trained / "artifacts/selected.joblib")


@pytest.fixture
def payload(frame):
    return Transaction.model_validate(
        frame.drop(columns="fraud_label").iloc[-1].to_dict()
    ).model_dump(mode="json")


@pytest.fixture
def client(trained, tmp_path):
    with TestClient(
        create_app(
            trained / "artifacts/selected.joblib",
            ServiceSettings(demo_mode=True, database=tmp_path / "audit.db"),
        )
    ) as c:
        yield c
