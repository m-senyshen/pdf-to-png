import Stripe from "stripe";

export async function onRequestPost(context) {
  const { env, request } = context;
  const stripe = new Stripe(env.STRIPE_SECRET_KEY);

  const body = await request.json();
  const mapSessionId = body?.mapSessionId;

  if (!mapSessionId) {
    return new Response("Missing mapSessionId", { status: 400 });
  }

  const successUrl = `${env.APP_ORIGIN}/app.html?paid=1`;
  const cancelUrl = `${env.APP_ORIGIN}/app.html?canceled=1`;

  const checkout = await stripe.checkout.sessions.create({
    mode: "payment",
    line_items: [{ price: env.STRIPE_PRICE_ID, quantity: 1 }],
    success_url: successUrl,
    cancel_url: cancelUrl,

    // IMPORTANT: tie Stripe payment to your session id
    client_reference_id: mapSessionId,

    // Optional: helps receipts + customer lookup
    // customer_creation: "always",
  });

  // Store “created but not paid yet” record (optional but useful)
  await env.DB.prepare(
    `INSERT INTO entitlements (session_id, paid, stripe_checkout_session_id)
     VALUES (?, 0, ?)
     ON CONFLICT(session_id) DO UPDATE SET stripe_checkout_session_id=excluded.stripe_checkout_session_id`
  ).bind(mapSessionId, checkout.id).run();

  return Response.json({ url: checkout.url });
}
