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

        // Morning schedule (8 AM - 12 PM)
        schedules.push({
          doctorId,
          startTime: currentDate.clone().set({ hour: 8, minute: 0 }).toDate(),
          endTime: currentDate.clone().set({ hour: 12, minute: 0 }).toDate(),
          status: ScheduleStatus.AVAILABLE,
        });

        // Afternoon schedule (1 PM - 6 PM)
        schedules.push({
          doctorId,
          startTime: currentDate.clone().set({ hour: 13, minute: 0 }).toDate(),
          endTime: currentDate.clone().set({ hour: 18, minute: 0 }).toDate(),
          status: ScheduleStatus.AVAILABLE,
        });
      }
    }

    await this.prisma.schedule.createMany({ data: schedules });
    return schedules;
  }

  async findDoctorSchedules(doctorId: string, startDate: Date, endDate: Date) {
    return await this.prisma.schedule.findMany({
      where: {
        doctorId,
        startTime: {
          gte: startDate,
          lte: endDate,
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

    // Check if user has permission to update
    if (currentUser.role !== 'ADMIN' && schedule.doctorId !== currentUser.id) {
      throw new ForbiddenException(
        'You do not have permission to update this schedule',
      );
    }

    return await this.prisma.schedule.update({
      where: { id: scheduleId },
      data: updateScheduleDto,
    });
  }
}
