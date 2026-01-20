export async function onRequestGet(context) {
  const { request, env } = context;

  const STRIPE_SECRET_KEY = env.STRIPE_SECRET_KEY;
  if (!STRIPE_SECRET_KEY) {
    return new Response("Missing STRIPE_SECRET_KEY", { status: 500 });
  }

  const url = new URL(request.url);
  const sessionId = url.searchParams.get("session_id");

  if (!sessionId) {
    return new Response(JSON.stringify({ paid: false, error: "Missing session_id" }), {
      status: 400,
      headers: { "Content-Type": "application/json" }
    });
  }

  const resp = await fetch(`https://api.stripe.com/v1/checkout/sessions/${encodeURIComponent(sessionId)}`, {
    headers: {
      "Authorization": `Bearer ${STRIPE_SECRET_KEY}`
    }
  });

  const data = await resp.json();

  if (!resp.ok) {
    return new Response(JSON.stringify({ paid: false, error: data }), {
      status: resp.status,
      headers: { "Content-Type": "application/json" }
    });
  }

  // Stripe returns payment_status: 'paid' when successful
  const paid = data.payment_status === "paid";

  return new Response(JSON.stringify({ paid }), {
    status: 200,
    headers: {
      "Content-Type": "application/json",
      "Access-Control-Allow-Origin": "*"
    }
  });
}
