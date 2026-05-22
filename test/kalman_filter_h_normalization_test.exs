defmodule BstsNx.KalmanFilterHNormalizationTest do
  use ExUnit.Case, async: true

  alias BstsNx.KalmanFilter

  @source_path Path.expand("../lib/bsts_nx/kalman_filter.ex", __DIR__)
  @external_resource @source_path

  test "normalize_h_series avoids slice calls for time-varying tensor H setup" do
    ast = @source_path |> File.read!() |> Code.string_to_quoted!()
    body = private_function_body!(ast, :normalize_h_series, 2)
    calls = remote_calls(body)

    refute MapSet.member?(calls, {Nx, :slice})
    refute MapSet.member?(calls, {Nx, :slice_along_axis})
    assert MapSet.member?(calls, {Nx, :to_flat_list})
    assert MapSet.member?(calls, {Nx, :to_batched})
  end

  test "time-varying tensor H encodings match equivalent list encodings" do
    assert_filter_series_close(
      KalmanFilter.filter_with_pred(
        [1.0, 1.4, 1.1, 1.8],
        1.0,
        Nx.tensor([1.0, 0.9, 1.1, 1.0], type: {:f, 32}),
        0.1,
        0.5,
        0.0,
        1.0
      ),
      KalmanFilter.filter_with_pred(
        [1.0, 1.4, 1.1, 1.8],
        1.0,
        [1.0, 0.9, 1.1, 1.0],
        0.1,
        0.5,
        0.0,
        1.0
      )
    )

    row_h = [[1.0, 0.0], [1.0, 0.2], [1.0, -0.1], [1.0, 0.3]]

    assert_filter_series_close(
      KalmanFilter.filter_with_pred(
        [1.0, 1.4, 1.1, 1.8],
        Nx.tensor([[1.0, 1.0], [0.0, 1.0]]),
        Nx.tensor(row_h, type: {:f, 32}),
        Nx.tensor([[0.1, 0.0], [0.0, 0.02]]),
        0.5,
        Nx.tensor([0.0, 0.0]),
        Nx.eye(2)
      ),
      KalmanFilter.filter_with_pred(
        [1.0, 1.4, 1.1, 1.8],
        Nx.tensor([[1.0, 1.0], [0.0, 1.0]]),
        Enum.map(row_h, &Nx.tensor(&1, type: {:f, 32})),
        Nx.tensor([[0.1, 0.0], [0.0, 0.02]]),
        0.5,
        Nx.tensor([0.0, 0.0]),
        Nx.eye(2)
      )
    )

    matrix_h = [
      [[1.0, 0.0], [0.0, 1.0]],
      [[1.0, 0.1], [0.0, 1.0]],
      [[1.0, -0.1], [0.2, 1.0]]
    ]

    assert_filter_series_close(
      KalmanFilter.filter_with_pred(
        [Nx.tensor([1.0, 0.5]), Nx.tensor([1.2, 0.6]), Nx.tensor([1.1, 0.4])],
        Nx.eye(2),
        Nx.tensor(matrix_h, type: {:f, 32}),
        Nx.multiply(Nx.eye(2), 0.1),
        Nx.multiply(Nx.eye(2), 0.2),
        Nx.tensor([0.0, 0.0]),
        Nx.eye(2)
      ),
      KalmanFilter.filter_with_pred(
        [Nx.tensor([1.0, 0.5]), Nx.tensor([1.2, 0.6]), Nx.tensor([1.1, 0.4])],
        Nx.eye(2),
        Enum.map(matrix_h, &Nx.tensor(&1, type: {:f, 32})),
        Nx.multiply(Nx.eye(2), 0.1),
        Nx.multiply(Nx.eye(2), 0.2),
        Nx.tensor([0.0, 0.0]),
        Nx.eye(2)
      )
    )
  end

  defp assert_filter_series_close({filtered_a, predicted_a}, {filtered_b, predicted_b}) do
    assert length(filtered_a) == length(filtered_b)
    assert length(predicted_a) == length(predicted_b)

    assert_state_pairs_close(filtered_a, filtered_b)
    assert_state_pairs_close(predicted_a, predicted_b)
  end

  defp assert_state_pairs_close(left, right) do
    Enum.zip(left, right)
    |> Enum.each(fn {{x_a, p_a}, {x_b, p_b}} ->
      assert_tensor_close(x_a, x_b)
      assert_tensor_close(p_a, p_b)
    end)
  end

  defp assert_tensor_close(left, right) do
    assert Nx.shape(left) == Nx.shape(right)
    assert Nx.all_close(left, right, atol: 1.0e-8, rtol: 1.0e-8) |> Nx.to_number() == 1
  end

  defp private_function_body!(ast, name, arity) do
    {_ast, matches} =
      Macro.prewalk(ast, [], fn
        {:defp, _, [{^name, _, args}, [do: body]]} = node, acc
        when is_list(args) and length(args) == arity ->
          {node, [body | acc]}

        node, acc ->
          {node, acc}
      end)

    case matches do
      [body] ->
        body

      [] ->
        flunk("expected to find defp #{name}/#{arity}")

      _ ->
        flunk("expected exactly one defp #{name}/#{arity}")
    end
  end

  defp remote_calls(ast) do
    {_ast, calls} =
      Macro.prewalk(ast, MapSet.new(), fn
        {{:., _, [{:__aliases__, _, module_parts}, function]}, _, _args} = node, acc ->
          {node, MapSet.put(acc, {Module.concat(module_parts), function})}

        node, acc ->
          {node, acc}
      end)

    calls
  end
end
