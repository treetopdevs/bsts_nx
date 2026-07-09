defmodule BstsSite.Demos.DefaultCache do
  @moduledoc """
  Lazy, process-independent memoization for deterministic demo defaults.

  Every demo's default figure is computed from fixed seeds, so the result is
  identical for every visitor. First reader computes and stores; later
  readers (including the disconnected static render and the WebSocket mount
  of the same visit) get a `:persistent_term` read.

  Only cache values derived from compile-time constants — never anything
  derived from user params. A duplicate concurrent first-compute is benign
  (both writers store the same deterministic value).
  """

  @spec get(term(), (-> result)) :: result when result: var
  def get(key, fun) when is_function(fun, 0) do
    pt_key = {__MODULE__, key}

    case :persistent_term.get(pt_key, :__miss__) do
      :__miss__ ->
        value = fun.()
        :persistent_term.put(pt_key, value)
        value

      value ->
        value
    end
  end
end
