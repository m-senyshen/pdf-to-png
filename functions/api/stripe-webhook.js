import Stripe from "stripe";

export async function onRequestPost(context) {
  const { env, request } = context;
  const stripe = new Stripe(env.STRIPE_SECRET_KEY);

  const sig = request.headers.get("stripe-signature");
  const rawBody = await request.text();

  let event;
  try {
    event = stripe.webhooks.constructEvent(rawBody, sig, env.STRIPE_WEBHOOK_SECRET);
  } catch (err) {
    return new Response(`Webhook Error: ${err.message}`, { status: 400 });
  }

  // We care about successful Checkout payment
  if (event.type === "checkout.session.completed") {
    const session = event.data.object;

    const mapSessionId = session.client_reference_id;
    const checkoutSessionId = session.id;
    const paymentIntentId = session.payment_intent;
    const email = session.customer_details?.email || null;

    if (mapSessionId) {
      await env.DB.prepare(
        `INSERT INTO entitlements (session_id, paid, stripe_checkout_session_id, stripe_payment_intent_id, customer_email)
         VALUES (?, 1, ?, ?, ?)
         ON CONFLICT(session_id) DO UPDATE SET
           paid=1,
           stripe_checkout_session_id=excluded.stripe_checkout_session_id,
           stripe_payment_intent_id=excluded.stripe_payment_intent_id,
           customer_email=excluded.customer_email`
      ).bind(mapSessionId, checkoutSessionId, paymentIntentId, email).run();
    }
  }

  return new Response("ok", { status: 200 });
}
