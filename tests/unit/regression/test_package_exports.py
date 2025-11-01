def test_root_package_reexports_expected_symbols():
    import ai_dev_agent
    from ai_dev_agent.engine.react import ReactiveExecutor
    from ai_dev_agent.providers.llm import create_client as provider_create
    from ai_dev_agent.tools.code.code_edit.tree_sitter_analysis import TreeSitterProjectAnalyzer

    assert isinstance(ai_dev_agent.__version__, str)
    assert ai_dev_agent.TreeSitterProjectAnalyzer is TreeSitterProjectAnalyzer
    assert ai_dev_agent.ReactiveExecutor is ReactiveExecutor
    assert ai_dev_agent.create_client is provider_create
    assert ai_dev_agent.core.configure_logging is ai_dev_agent.configure_logging


def test_core_and_providers_exports_align_with_implementations():
    from ai_dev_agent.core import write_artifact
    from ai_dev_agent.core.utils.artifacts import write_artifact as impl_write_artifact
    from ai_dev_agent.providers import create_client
    from ai_dev_agent.providers.llm import create_client as impl_create_client

    assert write_artifact is impl_write_artifact
    assert create_client is impl_create_client
    assert hasattr(create_client, "__call__")
