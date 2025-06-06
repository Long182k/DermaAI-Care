import {
  ConflictException,
  ForbiddenException,
  Injectable,
  NotFoundException,
} from '@nestjs/common';
import { ScheduleStatus, User } from '@prisma/client';
import { PrismaService } from '../prisma.service';
import { CreateAppointmentDto } from './dto/create-appointment.dto';
import { UpdateAppointmentDto } from './dto/update-appointment.dto';
import { MailerService } from '@nestjs-modules/mailer';

@Injectable()
export class AppointmentService {
  constructor(
    private prisma: PrismaService,
    private readonly mailerService: MailerService,
  ) {}

  async createAppointment(
    createAppointmentDto: CreateAppointmentDto,
    userId: string,
  ) {
    const { scheduleId, notes, patientId } = createAppointmentDto;

    // Start transaction to ensure data consistency
    return await this.prisma.$transaction(async (tx) => {
      const currentUserRole = await tx.user.findUnique({
        where: { id: userId ?? patientId },
      });

      // Check if schedule is available
      const schedule = await tx.schedule.findUnique({
        where: { id: scheduleId },
        include: { Doctor: true },
      });

      if (!schedule) {
        throw new ForbiddenException('Schedule not found');
      }

      if (schedule.status !== ScheduleStatus.AVAILABLE) {
        throw new ConflictException('This time slot is not available');
      }

      // Update schedule status to BOOKED
      await tx.schedule.update({
        where: { id: scheduleId },
        data: { status: ScheduleStatus.BOOKED },
      });

      // const mappedPatientId =
      //   currentUserRole.role === 'PATIENT' ? userId : patientId;

      // Create appointment
      const appointment = await tx.appointment.create({
        data: {
          patientId,
          doctorId: schedule.doctorId,
          scheduleId,
          status: 'SCHEDULED',
          notes,
        },
        include: {
          Patient: true,
          Doctor: true,
          Schedule: true,
        },
      });
      // Create notification to users' email.
      await tx.notification.create({
        data: {
          userId: patientId,
          appointmentId: appointment.id,
          method: 'EMAIL',
        },
      });

      // You might also want to create a notification for the doctor
      await tx.notification.create({
        data: {
          userId: schedule.doctorId, // Notification for the doctor
          appointmentId: appointment.id,
          method: 'EMAIL',
        },
      });

      if (appointment) {
        await this.mailerService.sendMail({
          to: appointment.Patient.email,
          subject: `Appointment Reminder: Your Appointment with Dr. ${appointment.Doctor.userName}`,
          template: 'appointment-reminder',
          context: {
            patientName: appointment.Patient.userName,
            doctorName: appointment.Doctor.userName,
            appointmentStartTime: appointment.Schedule.startTime,
            appointmentEndTime: appointment.Schedule.endTime,
            notes: appointment.notes,
            doctorEmail: appointment.Doctor.email,
            doctorAvatar: appointment.Doctor.avatarUrl,
            year: new Date().getFullYear(),
          },
        });
      }

      return appointment;
    });
  }

  async findUserAppointments(userId: string, role: string) {
    if (role === 'PATIENT') {
      return await this.prisma.appointment.findMany({
        where: { patientId: userId },
        include: {
          Patient: true,
          Doctor: true,
          Schedule: true,
        },
        orderBy: {
          createdAt: 'desc',
        },
      });
    }

    if (role === 'DOCTOR') {
      return await this.prisma.appointment.findMany({
        where: { doctorId: userId },
        include: {
          Patient: true,
          Doctor: true,
          Schedule: true,
        },
        orderBy: {
          createdAt: 'desc',
        },
      });
    }
  }

  async findAppointmentHistory(userId: string, role: string) {
    // Define where clause based on user role
    const where =
      role === 'DOCTOR'
        ? {
            doctorId: userId,
            status: { in: ['COMPLETED', 'CANCELLED'] },
          }
        : {
            patientId: userId,
            status: { in: ['COMPLETED', 'CANCELLED'] },
          };

    if (role === 'PATIENT') {
      return await this.prisma.appointment.findMany({
        where,
        include: {
          Patient: true,
          Doctor: true,
          Schedule: true,
        },
        orderBy: {
          Schedule: {
            startTime: 'desc',
          },
        },
      });
    }

    if (role === 'DOCTOR') {
      return await this.prisma.appointment.findMany({
        where,
        include: {
          Patient: true,
          Doctor: true,
          Schedule: true,
        },
        orderBy: {
          Schedule: {
            startTime: 'desc',
          },
        },
      });
    }
  }

  async updateAppointment(
    id: string,
    updateAppointmentDto: UpdateAppointmentDto,
    currentUser: User,
  ) {
    const appointment = await this.prisma.appointment.findUnique({
      where: { id },
      include: { Schedule: true },
    });

    if (!appointment) {
      throw new ForbiddenException('Appointment not found');
    }

    const userId = currentUser.id ?? currentUser.userId;

    // Check permissions
    if (
      currentUser.role !== 'ADMIN' &&
      appointment.patientId !== userId &&
      appointment.doctorId !== userId
    ) {
      throw new ForbiddenException(
        'You do not have permission to update this appointment',
      );
    }

    return await this.prisma.appointment.update({
      where: { id },
      data: updateAppointmentDto,
      include: {
        Patient: true,
        Doctor: true,
        Schedule: true,
      },
    });
  }

  async cancelAppointment(id: string, currentUser: User) {
    const appointment = await this.prisma.appointment.findUnique({
      where: { id },
      include: { Schedule: true },
    });

    if (!appointment) {
      throw new ForbiddenException('Appointment not found');
    }
    const userId = currentUser.id ?? currentUser.userId;

    // Check permissions
    if (appointment.patientId !== userId && appointment.doctorId !== userId) {
      throw new ForbiddenException(
        'You do not have permission to cancel this appointment',
      );
    }

    return await this.prisma.$transaction([
      // Update appointment status
      this.prisma.appointment.update({
        where: { id },
        data: { status: 'CANCELLED' },
      }),
      // Free up the schedule
      this.prisma.schedule.update({
        where: { id: appointment.scheduleId },
        data: { status: ScheduleStatus.AVAILABLE },
      }),
    ]);
  }

  async findAppointmentById(id: string) {
    const appointment = await this.prisma.appointment.findUnique({
      where: { id },
      include: {
        Patient: true,
        Doctor: true,
        Schedule: true,
      },
    });

    if (!appointment) {
      throw new ForbiddenException('Appointment not found');
    }

    return appointment;
  }

  async changeAppointmentStatus(
    appointmentId: string,
    status: string,
    currentUser: User,
  ): Promise<any> {
    // Find the appointment
    const appointment = await this.prisma.appointment.findUnique({
      where: { id: appointmentId },
      include: {
        Patient: true,
        Doctor: true,
      },
    });

    if (!appointment) {
      throw new NotFoundException('Appointment not found');
    }

    // Check permissions (only the admin can change status)
    const isAdmin = currentUser.role === 'ADMIN';

    if (!isAdmin) {
      throw new ForbiddenException(
        'You do not have permission to change this appointment status',
      );
    }

    // Additional validation based on roles and status transitions
    if (status === 'CONFIRMED' && !isAdmin) {
      throw new ForbiddenException(
        'Only doctors or admins can confirm appointments',
      );
    }

    if (status === 'COMPLETED' && !isAdmin) {
      throw new ForbiddenException(
        'Only doctors or admins can mark appointments as completed',
      );
    }

    // Update the appointment status
    const updatedAppointment = await this.prisma.appointment.update({
      where: { id: appointmentId },
      data: { status },
    });

    return updatedAppointment;
  }
}
