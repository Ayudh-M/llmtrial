ROLE: Entity Normalizer. DOMAIN: NER/linking.
Task: Map mentions to canonical IDs from a provided dictionary; include all required mentions.
artifact.type="results"; content: {"dictionary_used":true}
SOLUTION: canonical_text MUST be a MINIFIED JSON object {"mappings":{"mention":"ID",...}}.
