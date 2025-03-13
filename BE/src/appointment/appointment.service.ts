import {
  ConflictException,
  ForbiddenException,
  Injectable,
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
        where: { id: userId },
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

      const mappedPatientId =
        currentUserRole.role === 'PATIENT' ? userId : patientId;

      // Create appointment
      const appointment = await tx.appointment.create({
        data: {
          patientId: mappedPatientId,
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
      // Create reminder
      await tx.reminder.create({
        data: {
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
    const where =
      role === 'DOCTOR' ? { doctorId: userId } : { patientId: userId };

    return await this.prisma.appointment.findMany({
      where,
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

    // Check permissions
    if (
      currentUser.role !== 'ADMIN' &&
      appointment.patientId !== currentUser.id &&
      appointment.doctorId !== currentUser.id
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

    // Check permissions
    if (
      currentUser.role !== 'ADMIN' &&
      appointment.patientId !== currentUser.id &&
      appointment.doctorId !== currentUser.id
    ) {
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
}
