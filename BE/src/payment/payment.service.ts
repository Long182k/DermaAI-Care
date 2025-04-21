import { BadRequestException, Injectable } from '@nestjs/common';
import Stripe from 'stripe';
import { PrismaService } from '../prisma.service';

@Injectable()
export class PaymentService {
  private stripe: Stripe;

  constructor(
    private prisma: PrismaService,
  ) {
    this.stripe = new Stripe(
      process.env.STRIPE_SECRET_KEY
    );
  }

  async createCheckoutSession(appointmentId: string, userId: string) {
    // Get appointment details
    const appointment = await this.prisma.appointment.findUnique({
      where: { id: appointmentId },
      include: {
        Patient: true,
        Doctor: true,
        Schedule: true,
      },
    });

    if (!appointment) {
      throw new Error('Appointment not found');
    }

    // Check if payment already exists for this appointment
    const existingPayment = await this.prisma.payment.findFirst({
      where: { 
        appointmentId,
        status: 'COMPLETED' 
      },
    });
    
    if (existingPayment) {
      throw new BadRequestException('Payment already completed for this appointment');
    }

    // Create a checkout session
    const session = await this.stripe.checkout.sessions.create({
      payment_method_types: ['card'],
      line_items: [
        {
          price_data: {
            currency: 'usd',
            product_data: {
              name: `Appointment with Dr. ${appointment.Doctor.userName}`,
              description: `Medical consultation on ${new Date(
                appointment.Schedule.startTime,
              ).toLocaleString()}`,
            },
            unit_amount: 2000, // $20.00 in cents
          },
          quantity: 1,
        },
      ],
      mode: 'payment',
      success_url: `${process.env.FRONTEND_URL}/appointments`,
      cancel_url: `${process.env.FRONTEND_URL}/appointments`,
      metadata: {
        appointmentId: appointmentId,
        userId: userId,
      },
    });

    return { sessionId: session.id, url: session.url };
  }

  async handleWebhookEvent(event: Stripe.Event) {
    if (event.type === 'checkout.session.completed') {
      const session = event.data.object as Stripe.Checkout.Session;
      
      // Extract metadata from the session
      const { appointmentId, userId } = session.metadata;
      
      // Check if payment already exists with this transaction ID
      const existingPayment = await this.prisma.payment.findUnique({
        where: { transactionId: session.id },
      });
      
      if (!existingPayment) {
        // Create payment record only after successful checkout
        await this.prisma.payment.create({
          data: {
            userId,
            appointmentId,
            amount: session.amount_total / 100, // Convert from cents to dollars
            currency: session.currency.toUpperCase(),
            status: 'COMPLETED',
            paymentMethod: 'card',
            transactionId: session.id
          },
        });
        
        // Update appointment status to CONFIRMED or any appropriate status
        await this.prisma.appointment.update({
          where: { id: appointmentId },
          data: { status: 'CONFIRMED' }
        });
      }
    }
  }

  async getPaymentStatus(sessionId: string) {
    const payment = await this.prisma.payment.findUnique({
      where: { transactionId: sessionId },
    });
    
    return payment;
  }
}
