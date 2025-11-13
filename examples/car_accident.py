"""
Demonstration of the hierarchical reasoning pipeline for a car accident scenario.
"""

from __future__ import annotations

from typing import Iterable, Tuple

from layers.abstract import ConclusionRule
from layers.inference import InferenceRule
from models.hierarchical_reasoner import HierarchicalReasoner, ReasoningResult, build_reasoner
from models.types import AbstractConclusion, Fact, FactType, Relation, ensure_tuple


def causal_chain_rule(facts: Tuple[Fact, ...]) -> Iterable[Relation]:
    """
    Link facts that explicitly declare a causal dependency in their metadata.
    """

    index = {fact.label: fact for fact in facts}
    for fact in facts:
        cause_label = fact.metadata.get("caused_by")
        if cause_label:
            cause = index.get(cause_label)
            if cause:
                yield Relation(
                    subject=cause,
                    predicate="caused",
                    obj=fact,
                    weight=0.9,
                )
        impact_label = fact.metadata.get("impacted")
        if impact_label:
            impacted = index.get(impact_label)
            if impacted:
                yield Relation(
                    subject=fact,
                    predicate="impacted",
                    obj=impacted,
                    weight=0.7,
                )


def obligation_breach_rule(facts: Tuple[Fact, ...]) -> Iterable[Relation]:
    """
    Identify entities that violated domain-specific obligations.
    """

    index = {fact.label: fact for fact in facts}
    for fact in facts:
        breached = fact.metadata.get("breached_obligation")
        if breached:
            obligation = index.get(breached)
            if obligation:
                yield Relation(
                    subject=fact,
                    predicate="breached",
                    obj=obligation,
                    weight=1.0,
                )


def legal_liability_rule(
    facts: Tuple[Fact, ...],
    relations: Tuple[Relation, ...],
) -> Iterable[AbstractConclusion]:
    """
    Derive legal liability conclusions based on causal links and breaches.
    """

    severe_injuries = tuple(
        fact for fact in facts if fact.metadata.get("severity") in {"major", "critical"}
    )
    caused_relations = [
        relation for relation in relations if relation.predicate == "caused"
    ]
    breached_relations = [
        relation for relation in relations if relation.predicate == "breached"
    ]

    for relation in caused_relations:
        actor = relation.subject
        obligation = next(
            (rel.obj for rel in breached_relations if rel.subject == actor),
            None,
        )
        if not obligation or not severe_injuries:
            continue

        supporting = tuple(
            rel
            for rel in relations
            if rel.subject == actor
            or rel.obj == actor
            or rel.obj in severe_injuries
        )
        victims = sorted(
            {
                victim
                for fact in severe_injuries
                if (victim := fact.metadata.get("victim")) is not None
            }
        )
        victim_phrase = ", ".join(victims) if victims else "affected parties"
        summary = (
            f"{actor.label} bears primary liability for the collision due to "
            f"breaching {obligation.label} and causing injury to {victim_phrase}."
        )

        yield AbstractConclusion(
            theme="legal",
            summary=summary,
            supporting_relations=supporting,
            recommended_actions=ensure_tuple(
                [
                    "Initiate insurance liability assessment",
                    "Notify authorities of traffic code breach",
                ]
            ),
        )


def ethical_response_rule(
    facts: Tuple[Fact, ...],
    relations: Tuple[Relation, ...],
) -> Iterable[AbstractConclusion]:
    """
    Recommend ethical follow-ups based on injury severity and cooperation.
    """

    injury_facts = [
        fact for fact in facts if fact.metadata.get("severity") in {"major", "critical"}
    ]
    cooperative_relations = [
        relation
        for relation in relations
        if relation.predicate == "impacted" and relation.obj in injury_facts
    ]

    if not injury_facts:
        return ()

    responders = {
        relation.subject.label: relation.subject for relation in cooperative_relations
    }
    summary = (
        "Prioritize medical support for injured parties and provide psychological "
        "assistance to all involved participants."
    )

    if responders:
        responder_labels = ", ".join(sorted(responders))
        summary += f" Coordinate follow-up interviews with {responder_labels}."

    yield AbstractConclusion(
        theme="ethical",
        summary=summary,
        supporting_relations=tuple(cooperative_relations),
        recommended_actions=ensure_tuple(
            [
                "Deploy emergency medical services",
                "Offer counseling resources",
                "Facilitate community support outreach",
            ]
        ),
    )


def demo_car_accident() -> ReasoningResult:
    """
    Execute the reasoning pipeline for a car accident scenario.
    """

    reasoner: HierarchicalReasoner = build_reasoner(
        inference_rules=(
            causal_chain_rule,
            obligation_breach_rule,
        ),
        conclusion_rules=(
            legal_liability_rule,
            ethical_response_rule,
        ),
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

    return reasoner.analyze(observations)


if __name__ == "__main__":
    result = demo_car_accident()
    print("Facts:")
    for fact in result.facts:
        print(f"  - {fact.label}: {fact.description} ({fact.fact_type.name})")
    print("\nRelations:")
    for relation in result.relations:
        print(
            f"  - {relation.subject.label} {relation.predicate} {relation.obj.label}"
            f" [weight={relation.weight}]"
        )
    print("\nConclusions:")
    for conclusion in result.conclusions:
        print(f"  - [{conclusion.theme}] {conclusion.summary}")

