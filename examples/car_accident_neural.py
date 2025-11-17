"""
Neural network demonstration of the hierarchical reasoning pipeline for a car accident scenario.
"""

from __future__ import annotations

from models.neural.hierarchical_neural import build_neural_reasoner
from models.types import FactType


def demo_car_accident_neural():
    """
    Execute the neural reasoning pipeline for a car accident scenario.
    """
    reasoner = build_neural_reasoner(
        embedding_dim=256,
        hidden_dim=128,
        num_predicate_classes=10,
        num_theme_classes=5,
        predicate_vocab=["caused", "impacted", "breached", "violated", "affected"],
        theme_vocab=["legal", "ethical", "policy", "safety", "compliance"],
    )

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
            "metadata": {
                "actor_role": "driver_b",
            },
        },
        {
            "label": "speed_limit_law",
            "description": "Posted 35 mph limit for the intersection.",
            "type": FactType.CONTEXT.name,
            "metadata": {
                "enforced_on": "vehicle_a",
            },
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
        {
            "label": "injury_report",
            "description": "Driver B sustained a broken arm and concussion.",
            "type": FactType.EVENT.name,
            "metadata": {
                "caused_by": "collision_event",
                "severity": "major",
                "victim": "driver_b",
            },
        },
        {
            "label": "witness_statement",
            "description": "Witness reported Driver A running a red light.",
            "type": FactType.CONTEXT.name,
            "metadata": {"supports": "vehicle_b"},
        },
    ]

    result = reasoner.analyze(observations, reconstruct_objects=True)

    return result


if __name__ == "__main__":
    result = demo_car_accident_neural()

    print("Neural Reasoning Results:\n")
    print("=" * 60)

    print("\nFacts:")
    if result.facts:
        for fact in result.facts:
            print(f"  - {fact.label}: {fact.description} ({fact.fact_type.name})")
    else:
        print("  (Facts reconstructed from embeddings)")

    print("\nFact Embeddings:")
    for i, fe in enumerate(result.fact_embeddings):
        print(f"  - {fe.label}: embedding_dim={fe.embedding.shape[0]}, type_idx={fe.fact_type_idx}")

    print("\nRelations:")
    if result.relations:
        for relation in result.relations:
            print(
                f"  - {relation.subject.label} {relation.predicate} {relation.obj.label}"
                f" [weight={relation.weight:.3f}]"
            )
    else:
        print("  (Relations reconstructed from embeddings)")

    print("\nRelation Embeddings:")
    for i, re in enumerate(result.relation_embeddings):
        print(
            f"  - {re.predicate_label}: subject_idx={re.subject_idx}, obj_idx={re.obj_idx}, "
            f"weight={re.weight:.3f}"
        )

    print("\nConclusions:")
    if result.conclusions:
        for conclusion in result.conclusions:
            print(f"  - [{conclusion.theme}] {conclusion.summary}")
            if conclusion.supporting_relations:
                print(f"    Supporting relations: {len(conclusion.supporting_relations)}")
    else:
        print("  (Conclusions reconstructed from embeddings)")

    print("\nConclusion Embeddings:")
    for i, ce in enumerate(result.conclusion_embeddings):
        print(
            f"  - {ce.theme_label}: summary_dim={ce.summary_embedding.shape[0]}, "
            f"supporting_relations={len(ce.supporting_relation_indices)}"
        )

    print("\n" + "=" * 60)

