export async function onRequestGet(context) {
  const { env, request } = context;
  const url = new URL(request.url);
  const mapSessionId = url.searchParams.get("mapSessionId");

  if (!mapSessionId) return Response.json({ paid: false });

  const row = await env.DB.prepare(
    "SELECT paid FROM entitlements WHERE session_id = ?"
  ).bind(mapSessionId).first();

  return Response.json({ paid: row?.paid === 1 });
}
