defmodule BstsNx.Synthetic.AdstockTest do
  use ExUnit.Case, async: true

  alias BstsNx.Synthetic.Adstock

  describe "geometric_adstock/2" do
    test "empty signal returns empty" do
      assert Adstock.geometric_adstock([], 0.5) == []
    end

    test "lambda=0 returns original signal (no carry-over)" do
      signal = [1.0, 0.0, 0.0, 0.0, 0.0]
      assert Adstock.geometric_adstock(signal, 0.0) == signal
    end

    test "single impulse decays geometrically" do
      signal = [1.0, 0.0, 0.0, 0.0]
      result = Adstock.geometric_adstock(signal, 0.5)
      assert result == [1.0, 0.5, 0.25, 0.125]
    end

    test "lambda=1 accumulates indefinitely" do
      signal = [1.0, 0.0, 0.0, 0.0]
      result = Adstock.geometric_adstock(signal, 1.0)
      assert result == [1.0, 1.0, 1.0, 1.0]
    end

    test "multiple impulses accumulate" do
      signal = [1.0, 0.0, 1.0, 0.0, 0.0]
      result = Adstock.geometric_adstock(signal, 0.5)
      # t=0: 1.0
      # t=1: 0.0 + 0.5 * 1.0 = 0.5
      # t=2: 1.0 + 0.5 * 0.5 = 1.25
      # t=3: 0.0 + 0.5 * 1.25 = 0.625
      # t=4: 0.0 + 0.5 * 0.625 = 0.3125
      assert result == [1.0, 0.5, 1.25, 0.625, 0.3125]
    end

    test "hand-computed 5-element sequence with lambda=0.7" do
      signal = [2.0, 0.0, 3.0, 0.0, 1.0]
      result = Adstock.geometric_adstock(signal, 0.7)
      # t=0: 2.0
      # t=1: 0.0 + 0.7 * 2.0 = 1.4
      # t=2: 3.0 + 0.7 * 1.4 = 3.98
      # t=3: 0.0 + 0.7 * 3.98 = 2.786
      # t=4: 1.0 + 0.7 * 2.786 = 2.9502
      assert_in_delta Enum.at(result, 0), 2.0, 1.0e-10
      assert_in_delta Enum.at(result, 1), 1.4, 1.0e-10
      assert_in_delta Enum.at(result, 2), 3.98, 1.0e-10
      assert_in_delta Enum.at(result, 3), 2.786, 1.0e-10
      assert_in_delta Enum.at(result, 4), 2.9502, 1.0e-10
    end
  end

  describe "hill_saturation/3" do
    test "empty signal returns empty" do
      assert Adstock.hill_saturation([], 1.0, 1.0) == []
    end

    test "zero input returns zero" do
      assert Adstock.hill_saturation([0.0], 1.0, 1.0) == [0.0]
    end

    test "x=ec returns 0.5 (half-saturation point)" do
      [result] = Adstock.hill_saturation([5.0], 5.0, 1.0)
      assert_in_delta result, 0.5, 1.0e-10
    end

    test "x=ec returns 0.5 for any slope" do
      for slope <- [0.5, 1.0, 2.0, 5.0] do
        [result] = Adstock.hill_saturation([3.0], 3.0, slope)
        assert_in_delta result, 0.5, 1.0e-10
      end
    end

    test "large x approaches 1.0" do
      [result] = Adstock.hill_saturation([1000.0], 1.0, 1.0)
      assert result > 0.999
    end

    test "small x approaches 0.0" do
      [result] = Adstock.hill_saturation([0.001], 1.0, 1.0)
      assert result < 0.002
    end

    test "negative values return 0.0" do
      result = Adstock.hill_saturation([-1.0, -0.5, 0.0], 1.0, 1.0)
      assert result == [0.0, 0.0, 0.0]
    end

    test "steeper slope gives sharper transition" do
      # With steep slope, values below ec should be closer to 0
      # and values above ec should be closer to 1
      signal = [0.5, 2.0]
      ec = 1.0

      [low_gentle, high_gentle] = Adstock.hill_saturation(signal, ec, 1.0)
      [low_steep, high_steep] = Adstock.hill_saturation(signal, ec, 5.0)

      # Steeper slope means lower values below ec, higher values above ec
      assert low_steep < low_gentle
      assert high_steep > high_gentle
    end

    test "hand-computed values: ec=2.0, slope=3.0" do
      signal = [1.0, 2.0, 4.0]
      result = Adstock.hill_saturation(signal, 2.0, 3.0)
      # H(1) = 1 / (1 + (1/2)^(-3)) = 1 / (1 + 8) = 1/9
      # H(2) = 1 / (1 + (2/2)^(-3)) = 1 / (1 + 1) = 0.5
      # H(4) = 1 / (1 + (4/2)^(-3)) = 1 / (1 + 0.125) = 1/1.125
      assert_in_delta Enum.at(result, 0), 1.0 / 9.0, 1.0e-10
      assert_in_delta Enum.at(result, 1), 0.5, 1.0e-10
      assert_in_delta Enum.at(result, 2), 1.0 / 1.125, 1.0e-10
    end
  end
end
