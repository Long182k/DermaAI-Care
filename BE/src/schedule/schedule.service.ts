import { ForbiddenException, Injectable } from '@nestjs/common';
import { ScheduleStatus, User } from '@prisma/client';
import { default as moment } from 'moment';
import { PrismaService } from '../prisma.service';
import { CreateScheduleDto } from './dto/create-schedule.dto';
import { UpdateScheduleDto } from './dto/update-schedule.dto';

@Injectable()
export class ScheduleService {
  constructor(private prisma: PrismaService) {}

  async generateDefaultSchedule(doctorId: string) {
    const schedules: CreateScheduleDto[] = [];
    const startDate = moment().startOf('week');

    // Generate for next 4 weeks
    for (let week = 0; week < 4; week++) {
      // Monday to Friday
      for (let day = 1; day <= 5; day++) {
        const currentDate = moment(startDate)
          .add(week, 'weeks')
          .add(day - 1, 'days');

        // Morning schedule (8 AM - 12 PM) with 30-minute slots
        for (let hour = 8; hour < 12; hour++) {
          for (let minute = 0; minute < 60; minute += 30) {
            const slotStart = currentDate
              .clone()
              .set({ hour, minute, second: 0, millisecond: 0 });

            const slotEnd = slotStart.clone().add(30, 'minutes');

            schedules.push({
              doctorId,
              startTime: slotStart.toISOString(),
              endTime: slotEnd.toISOString(),
              status: ScheduleStatus.AVAILABLE,
            });
          }
        }

        // Afternoon schedule (1 PM - 6 PM) with 30-minute slots
        for (let hour = 13; hour < 18; hour++) {
          for (let minute = 0; minute < 60; minute += 30) {
            const slotStart = currentDate
              .clone()
              .set({ hour, minute, second: 0, millisecond: 0 });

            const slotEnd = slotStart.clone().add(30, 'minutes');

            schedules.push({
              doctorId,
              startTime: slotStart.toISOString(),
              endTime: slotEnd.toISOString(),
              status: ScheduleStatus.AVAILABLE,
            });
          }
        }
      }
    }

    await this.prisma.schedule.createMany({ data: schedules });
    return schedules;
  }

  async findDoctorSchedules(
    doctorId: string,
    startDate: Date,
    endDate: Date,
    filter?: 'day' | 'week' | 'month',
  ) {
    let adjustedStartDate = new Date(startDate);
    let adjustedEndDate = new Date(endDate);

    // Apply filter if provided
    if (filter) {
      const start = moment(startDate);

      switch (filter) {
        case 'day':
          adjustedStartDate = start.startOf('day').toDate();
          adjustedEndDate = start.endOf('day').toDate();
          break;
        case 'week':
          adjustedStartDate = start.startOf('week').toDate();
          adjustedEndDate = start.endOf('week').toDate();
          break;
        case 'month':
          adjustedStartDate = start.startOf('month').toDate();
          adjustedEndDate = start.endOf('month').toDate();
          break;
      }
    }

    return await this.prisma.schedule.findMany({
      where: {
        doctorId,
        startTime: {
          gte: adjustedStartDate.toISOString(),
          lte: adjustedEndDate.toISOString(),
        },
      },
      orderBy: {
        startTime: 'asc',
      },
    });
  }

  async updateSchedule(
    scheduleId: string,
    updateScheduleDto: UpdateScheduleDto,
    currentUser: User,
  ) {
    const schedule = await this.prisma.schedule.findUnique({
      where: { id: scheduleId },
      include: { Doctor: true },
    });

    if (!schedule) {
      throw new ForbiddenException('Schedule not found');
    }
    const userId = currentUser.id ?? currentUser.userId;
    // Check if user has permission to update
    if (currentUser.role !== 'ADMIN' && schedule.doctorId !== userId) {
      throw new ForbiddenException(
        'You do not have permission to update this schedule',
      );
    }

    const { startTime, endTime } = updateScheduleDto;

    const updatedData: UpdateScheduleDto = {
      startTime: new Date(startTime).toISOString(),
      endTime: new Date(endTime).toISOString(),
    };

    return await this.prisma.schedule.update({
      where: { id: scheduleId },
      data: updatedData,
    });
  }
}
