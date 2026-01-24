export async function onRequestGet({ request, env }) {
  if (!env.DB) return new Response("Missing D1 binding env.DB", { status: 500 });

  const url = new URL(request.url);
  const mapSessionId = url.searchParams.get("session_id");
  if (!mapSessionId) return new Response("Missing session_id", { status: 400 });

  const row = await env.DB.prepare(
    "SELECT paid FROM entitlements WHERE session_id = ?"
  )
    .bind(mapSessionId)
    .first();

  return new Response(JSON.stringify({ paid: !!row, paid_at: row?.paid_at || null }), {
    headers: { "Content-Type": "application/json" },
  });
}
