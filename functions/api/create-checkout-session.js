import Stripe from "stripe";

export async function onRequestPost({ request, env }) {
  try {
    if (!env?.STRIPE_SECRET_KEY) {
      return new Response("Missing env.STRIPE_SECRET_KEY", { status: 500 });
    }

    // Build origin safely from the actual request URL
    const origin = new URL(request.url).origin;

    const stripe = new Stripe(env.STRIPE_SECRET_KEY, {
      apiVersion: "2024-06-20", // ok to omit, but nice to pin
    });

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
            unit_amount: 1200, // $12.00 CAD
          },
          quantity: 1,
        },
      ],
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
