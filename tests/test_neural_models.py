"""
Unit tests for neural model components.
"""

import pytest
import torch

from layers.sensory import Observation
from models.neural.abstract_reasoner import AbstractReasoner
from models.neural.hierarchical_neural import HierarchicalNeuralReasoner, build_neural_reasoner
from models.neural.relation_network import RelationNetwork
from models.neural.sensory_encoder import SensoryEncoder
from models.types import FactType


@pytest.fixture
def device():
    """Fixture for device."""
    return torch.device("cpu")


@pytest.fixture
def sample_observations():
    """Fixture for sample observations."""
    return [
        {
            "label": "vehicle_a",
            "description": "Driver A's sedan traveling northbound.",
            "type": FactType.OBJECT.name,
            "metadata": {"actor_role": "driver_a"},
        },
        {
            "label": "collision_event",
            "description": "The two vehicles collided in the intersection.",
            "type": FactType.EVENT.name,
            "metadata": {"caused_by": "vehicle_a"},
        },
    ]


class TestSensoryEncoder:
    """Tests for SensoryEncoder."""

    def test_init(self, device):
        """Test encoder initialization."""
        encoder = SensoryEncoder(embedding_dim=128)
        encoder.to(device)
        assert encoder.embedding_dim == 128

    def test_forward(self, device, sample_observations):
        """Test forward pass."""
        encoder = SensoryEncoder(embedding_dim=128)
        encoder.to(device)
        encoder.eval()

        with torch.no_grad():
            fact_embeddings, fact_type_indices, metadata_embeddings = encoder.forward(
                sample_observations
            )

        assert fact_embeddings.shape[0] == len(sample_observations)
        assert fact_embeddings.shape[1] == 128
        assert fact_type_indices.shape[0] == len(sample_observations)

    def test_encode_facts(self, device, sample_observations):
        """Test fact encoding."""
        encoder = SensoryEncoder(embedding_dim=128)
        encoder.to(device)
        encoder.eval()

        with torch.no_grad():
            fact_embeddings = encoder.encode_facts(sample_observations)

        assert len(fact_embeddings) == len(sample_observations)
        assert all(fe.embedding.shape[0] == 128 for fe in fact_embeddings)


class TestRelationNetwork:
    """Tests for RelationNetwork."""

    def test_init(self, device):
        """Test network initialization."""
        network = RelationNetwork(embedding_dim=128, num_predicate_classes=5)
        network.to(device)
        assert network.embedding_dim == 128
        assert network.num_predicate_classes == 5

    def test_forward(self, device):
        """Test forward pass."""
        network = RelationNetwork(embedding_dim=128, num_predicate_classes=5)
        network.to(device)
        network.eval()

        num_facts = 3
        fact_embeddings = torch.randn(num_facts, 128)

        with torch.no_grad():
            predicate_logits, relation_weights = network.forward(fact_embeddings)

        num_edges = num_facts * (num_facts - 1)
        assert predicate_logits.shape[0] == num_edges
        assert predicate_logits.shape[1] == 5
        assert relation_weights.shape[0] == num_edges

    def test_predict_relations(self, device):
        """Test relation prediction."""
        from models.types import FactEmbedding

        network = RelationNetwork(embedding_dim=128, num_predicate_classes=5)
        network.to(device)
        network.eval()

        fact_embeddings = [
            FactEmbedding(
                embedding=torch.randn(128),
                fact_type_idx=0,
                label="fact_1",
            ),
            FactEmbedding(
                embedding=torch.randn(128),
                fact_type_idx=1,
                label="fact_2",
            ),
        ]

        with torch.no_grad():
            relations = network.predict_relations(
                fact_embeddings,
                predicate_vocab=["caused", "impacted"],
                threshold=0.3,
            )

        assert len(relations) > 0
        assert all(0 <= rel.weight <= 1 for rel in relations)


class TestAbstractReasoner:
    """Tests for AbstractReasoner."""

    def test_init(self, device):
        """Test reasoner initialization."""
        reasoner = AbstractReasoner(
            fact_embedding_dim=128,
            relation_embedding_dim=5,
            num_theme_classes=3,
        )
        reasoner.to(device)
        assert reasoner.hidden_dim == 256
        assert reasoner.num_theme_classes == 3

    def test_forward(self, device):
        """Test forward pass."""
        reasoner = AbstractReasoner(
            fact_embedding_dim=128,
            relation_embedding_dim=5,
            num_theme_classes=3,
        )
        reasoner.to(device)
        reasoner.eval()

        batch_size = 2
        num_facts = 3
        num_relations = 2

        fact_embeddings = torch.randn(batch_size, num_facts, 128)
        relation_embeddings = torch.randn(batch_size, num_relations, 5)

        with torch.no_grad():
            theme_logits, summary_embeddings, relation_scores = reasoner.forward(
                fact_embeddings,
                relation_embeddings,
            )

        assert theme_logits.shape == (batch_size, 3)
        assert summary_embeddings.shape[0] == batch_size
        assert relation_scores.shape == (batch_size, num_relations)

    def test_generate_conclusions(self, device):
        """Test conclusion generation."""
        from models.types import FactEmbedding, RelationEmbedding

        reasoner = AbstractReasoner(
            fact_embedding_dim=128,
            relation_embedding_dim=5,
            num_theme_classes=3,
        )
        reasoner.to(device)
        reasoner.eval()

        fact_embeddings = [
            FactEmbedding(
                embedding=torch.randn(128),
                fact_type_idx=0,
                label="fact_1",
            ),
        ]

        relation_embeddings = [
            RelationEmbedding(
                subject_idx=0,
                obj_idx=0,
                predicate_embedding=torch.randn(5),
                weight=0.8,
                predicate_label="caused",
            ),
        ]

        with torch.no_grad():
            conclusions = reasoner.generate_conclusions(
                fact_embeddings,
                relation_embeddings,
                theme_vocab=["legal", "ethical"],
            )

        assert len(conclusions) > 0


class TestHierarchicalNeuralReasoner:
    """Tests for HierarchicalNeuralReasoner."""

    def test_init(self, device):
        """Test reasoner initialization."""
        reasoner = build_neural_reasoner(
            embedding_dim=128,
            hidden_dim=64,
            num_predicate_classes=5,
            num_theme_classes=3,
            device=device,
        )
        assert reasoner.device == device

    def test_analyze(self, device, sample_observations):
        """Test full pipeline analysis."""
        reasoner = build_neural_reasoner(
            embedding_dim=128,
            hidden_dim=64,
            num_predicate_classes=5,
            num_theme_classes=3,
            device=device,
        )

        result = reasoner.analyze(
            sample_observations,
            relation_threshold=0.3,
            reconstruct_objects=True,
        )

        assert len(result.fact_embeddings) == len(sample_observations)
        assert result.facts is not None
        assert len(result.facts) == len(sample_observations)
        assert result.relations is not None
        assert result.conclusions is not None

    def test_train_eval_modes(self, device):
        """Test training and evaluation mode switching."""
        reasoner = build_neural_reasoner(device=device)

        reasoner.train()
        assert reasoner.sensory_encoder.training
        assert reasoner.relation_network.training
        assert reasoner.abstract_reasoner.training

        reasoner.eval()
        assert not reasoner.sensory_encoder.training
        assert not reasoner.relation_network.training
        assert not reasoner.abstract_reasoner.training


def test_car_accident_integration(device):
    """Integration test with car accident example."""
    from examples.car_accident import demo_car_accident

    rule_based_result = demo_car_accident()

    observations = [
        {
            "label": "vehicle_a",
            "description": "Driver A's sedan traveling northbound.",
            "type": FactType.OBJECT.name,
            "metadata": {
                "actor_role": "driver_a",
                "breached_obligation": "speed_limit_law",
            },
        },
        {
            "label": "vehicle_b",
            "description": "Driver B's crossover entering the intersection.",
            "type": FactType.OBJECT.name,
            "metadata": {"actor_role": "driver_b"},
        },
        {
            "label": "collision_event",
            "description": "The two vehicles collided in the intersection.",
            "type": FactType.EVENT.name,
            "metadata": {
                "caused_by": "vehicle_a",
                "impacted": "vehicle_b",
                "location": "5th Ave & Pine St",
            },
        },
    ]

    reasoner = build_neural_reasoner(
        embedding_dim=128,
        hidden_dim=64,
        num_predicate_classes=10,
        num_theme_classes=5,
        device=device,
        predicate_vocab=["caused", "impacted", "breached"],
        theme_vocab=["legal", "ethical", "policy"],
    )

    neural_result = reasoner.analyze(
        observations,
        relation_threshold=0.3,
        reconstruct_objects=True,
    )

    assert len(neural_result.fact_embeddings) == len(observations)
    assert neural_result.facts is not None
    assert neural_result.relations is not None
    assert neural_result.conclusions is not None

