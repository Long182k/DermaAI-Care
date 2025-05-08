import { BadRequestException, Injectable } from '@nestjs/common';
import Stripe from 'stripe';
import { PrismaService } from '../prisma.service';
import { MailerService } from '@nestjs-modules/mailer';

@Injectable()
export class PaymentService {
  private stripe: Stripe;

  constructor(
    private prisma: PrismaService,
    private readonly mailerService: MailerService,
  ) {
    this.stripe = new Stripe(process.env.STRIPE_SECRET_KEY);
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
        status: 'COMPLETED',
      },
    });

    if (existingPayment) {
      throw new BadRequestException(
        'Payment already completed for this appointment',
      );
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

    if (session) {
      await this.mailerService.sendMail({
        to: appointment.Patient.email,
        subject: `Payment Confirmation for Appointment with Dr. ${appointment.Doctor.userName}`,
        template: 'payment-confirmation',
        context: {
          patientName: appointment.Patient.userName,
          doctorName: appointment.Doctor.userName,
          amount: (session.amount_total / 100).toFixed(2),
          transactionId: session.id,
          paymentDate: new Date().toLocaleDateString(),
          appointmentStartTime: new Date(
            appointment.Schedule.startTime,
          ).toLocaleString(),
          appointmentEndTime: new Date(
            appointment.Schedule.endTime,
          ).toLocaleString(),
          appointmentUrl: `${process.env.FRONTEND_URL}/appointments`,
          doctorEmail: appointment.Doctor.email,
          doctorAvatar:
            appointment.Doctor.avatarUrl || 'https://via.placeholder.com/80',
          year: new Date().getFullYear(),
        },
      });

      return { sessionId: session.id, url: session.url };
    }
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
        await this.prisma.payment.create({
          data: {
            userId,
            appointmentId,
            amount: session.amount_total / 100, 
            currency: session.currency.toUpperCase(),
            status: 'COMPLETED',
            paymentMethod: 'VISA',
            transactionId: session.id,
          },
        });

        // Update appointment status to CONFIRMED or any appropriate status
        await this.prisma.appointment.update({
          where: { id: appointmentId },
          data: { status: 'CONFIRMED' },
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

  async getAppointments() {
    // Only include CONFIRMED appointments for "Appointments by Day"
    const confirmedAppointments = await this.prisma.appointment.findMany({
      where: { status: 'CONFIRMED' },
      select: {
        id: true,
        createdAt: true,
        status: true,
        Schedule: {
          select: { startTime: true }
        }
      },
    });

    // Group by day of week (e.g., Monday, Tuesday, ...)
    const daysOfWeek = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
    const appointmentsByDay = daysOfWeek.map((day) => ({
      day,
      count: 0,
    }));

    confirmedAppointments.forEach((appointment) => {
      const date = appointment.Schedule?.startTime
        ? new Date(appointment.Schedule.startTime)
        : appointment.createdAt
        ? new Date(appointment.createdAt)
        : null;
      if (date) {
        const dayIndex = date.getDay();
        appointmentsByDay[dayIndex].count += 1;
      }
    });

    return {
      summary: {
        // ...existing summary fields...
      },
      charts: {
        // ...existing chart fields...
        appointmentsByDay,
      },
    };
  }
}
