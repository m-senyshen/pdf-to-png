export async function onRequestGet({ request, env }) {
  const url = new URL(request.url);
  const map_session_id = url.searchParams.get("map_session_id");
  if (!map_session_id) return new Response("Missing map_session_id", { status: 400 });

  const row = await env.DB.prepare(
    "SELECT unlocked FROM entitlements WHERE map_session_id = ?"
  ).bind(map_session_id).first();

  const unlocked = row?.unlocked === 1;
  return new Response(JSON.stringify({ unlocked }), {
    headers: { "Content-Type": "application/json" },
  });
}
