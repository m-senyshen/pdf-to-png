import Stripe from "stripe";

export async function onRequestPost({ request, env }) {
  try {
    if (!env.STRIPE_SECRET_KEY) return new Response("Missing env.STRIPE_SECRET_KEY", { status: 500 });

    const { map_session_id } = await request.json().catch(() => ({}));
    if (!map_session_id) return new Response("Missing map_session_id", { status: 400 });

    const baseUrl = env.SITE_URL.replace(/\/+$/, "");

    const stripe = new Stripe(env.STRIPE_SECRET_KEY);

    const session = await stripe.checkout.sessions.create({
      mode: "payment",
      payment_method_types: ["card"],
      line_items: [
        {
          price_data: {
            currency: "cad",
            product_data: {
              name: "Map Extract – Point Export",
              description: "Export points from a georeferenced map",
            },
            unit_amount: 1200,
          },
          quantity: 1,
        },
      ],

      // tie payment to this browser session (useful for webhook/D1 entitlements)
      metadata: { map_session_id },
      const origin = new URL(request.url).origin;

      success_url: `${origin}/app?paid=1`,
      cancel_url: `${origin}/app?canceled=1`
      });

    return new Response(JSON.stringify({ url: session.url }), {
      headers: { "Content-Type": "application/json" },
    });
  } catch (err) {
    return new Response(`create-checkout-session error:\n${err?.message || String(err)}`, { status: 500 });
  }
}
