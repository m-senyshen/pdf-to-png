import Stripe from "stripe";

export async function onRequestPost({ request, env }) {
  try {
    if (!env.STRIPE_SECRET_KEY) {
      return new Response("Missing env.STRIPE_SECRET_KEY", { status: 500 });
    }

    const stripe = new Stripe(env.STRIPE_SECRET_KEY);

    const origin = new URL(request.url).origin;

    // Expect JSON body: { map_session_id: "..." }
    const body = await request.json().catch(() => ({}));
    const mapSessionId = body?.map_session_id;

    if (!mapSessionId) {
      return new Response("Missing map_session_id", { status: 400 });
    }

    const session = await stripe.checkout.sessions.create({
      mode: "payment",
      line_items: [
        {
          price_data: {
            currency: "cad",
            product_data: {
              name: "Map Extract – Point Export",
              description: "Export points (GeoJSON + CSV) from a georeferenced map session",
            },
            unit_amount: 1200, // $12.00 CAD
          },
          quantity: 1,
        },
      ],
      // Key part: tie payment to a session
      metadata: {
        map_session_id: mapSessionId,
      },

      success_url: `${origin}/app.html?paid=1`,
      cancel_url: `${origin}/app.html?canceled=1`,
    });

    return new Response(JSON.stringify({ url: session.url }), {
      headers: { "Content-Type": "application/json" },
    });
  } catch (err) {
    return new Response(
      `create-checkout-session error:\n${err?.message || String(err)}`,
      { status: 500 }
    );
  }
}
