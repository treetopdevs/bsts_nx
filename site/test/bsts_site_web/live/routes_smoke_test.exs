defmodule BstsSiteWeb.RoutesSmokeTest do
  use BstsSiteWeb.ConnCase, async: true

  test "GET / sends a content-security-policy header that restricts scripts to self", %{
    conn: conn
  } do
    conn = get(conn, ~p"/")

    [csp] = get_resp_header(conn, "content-security-policy")

    assert csp =~ "script-src 'self'"
  end
end
