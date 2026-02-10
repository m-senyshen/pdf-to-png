export async function onRequestGet({ request, env }) {
  if (!env.DB) return new Response("Missing D1 binding env.DB", { status: 500 });

  const url = new URL(request.url);
  const mapSessionId = url.searchParams.get("map_session_id");
  if (!mapSessionId) return new Response("Missing map_session_id", { status: 400 });

  const row = await env.DB.prepare(`
    SELECT
      paid,
      expires_at
    FROM entitlements
    WHERE map_session_id = ?
  `)
    .bind(mapSessionId)
    .first();

  return new Response(
    JSON.stringify({
      paid: row?.paid === 1 && (!row?.expires_at || new Date(row.expires_at) > new Date()),      expires_at: row?.expires_at ?? null,
      map_session_id: mapSessionId,
    }),
    {
      headers: {
        "Content-Type": "application/json",
        "Cache-Control": "no-store",
      },
    }
  );
}