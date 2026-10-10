<!-- Transcription of "ICD-10 to SNOWMEDCT/Screenshot 2026-03-17 at 4.51.58 AM.png" (NLM SNOMED CT US Edition
     map documentation page captured by the user). Used as an SDFT knowledge passage. -->
# SNOMED CT to ICD-10-CM map: release schedule and file path

**Release schedule.** The map is updated with each new SNOMED CT US Edition release (March and September) and includes the annual ICD-10-CM update.

**File path.** The SNOMED CT to ICD-10-CM map is released as refset 6011000124106, "ICD-10-CM complex map reference set (foundation metadata concept)". The map data are in the file `der2_iisssccRefset_ExtendedMap(Full|Snapshot)_US1000124_YYYYMMDD.txt`. That file is in the `Map` folder under `Refset` in each release-type folder (`Full/Refset/Map` or `Snapshot/Refset/Map`). The same folder also holds `der2_sRefset_SimpleMap…`.

**Columns.** id, effectiveTime, active, moduleId, refsetId, referencedComponentId (the SNOMED concept), mapGroup, mapPriority, mapRule, mapAdvice, mapTarget (the ICD-10-CM code), correlationId, mapCategoryId.
- Rules such as `TRUE` mean the mapping is unconditional.
- `IFA <conceptId> | <term> |` means the mapping is conditional on another concept.
- `OTHERWISE TRUE` is a fallback rule.
- Advice such as `ALWAYS K81.9`, `CONSIDER ADDITIONAL CODE TO IDENTIFY SPECIFIC CONDITION OR DISEASE`, `MAP OF SOURCE CONCEPT IS CONTEXT DEPENDENT` and `MAP SOURCE CONCEPT CANNOT BE CLASSIFIED WITH AVAILABLE DATA` explains how to apply the map.

The map is also available in a human-readable form, `tls_Icd10cmHumanReadableMap_US1000124_YYYYMMDD.tsv`, from the US Edition page.
