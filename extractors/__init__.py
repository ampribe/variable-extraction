"""
Included modules and classes:
    extractor_config: ExtractorConfig (configuration file for VariableExtractor)
    extractor_log: ExtractorLog (log file for variable extraction)
    variable_extractor: VariableExtractor (Base variable extraction class)
    VariableExtractor subclasses:
        bench_ruling_classifier: BenchRulingClassifier (classifies outcome of bench trial)
        jury_ruling_classifier: JuryRulingClassifier (classifies outcome of jury trial)
        trial_type_classifier: TrialTypeClassifier (classifies trial into bench, jury, unknown)
        document_summarizer: DocumentSummarizer (summarizes case events)
"""
